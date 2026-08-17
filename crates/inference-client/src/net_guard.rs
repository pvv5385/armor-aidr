//! Address validation for the sidecar endpoint.
//!
//! Armor's other network egress goes through the deterministic engine's own
//! request handling; nothing else in `armor-core` opens an outbound
//! connection to an address named by policy data, so nothing else needed
//! this kind of guard. This is the first client to resolve a
//! policy-controlled URL and dial it, hence the SSRF exposure below and the
//! guard that closes it.
//!
//! # Why a guard at all, for an internal service
//!
//! `Backend::endpoint_url` comes from **policy**. A policy is data: it arrives
//! from `config/policies.yaml`, from the Postgres control plane, or from a
//! synced bundle, and anyone who can write one can name any URL they like.
//! That makes this a server-side request forgery surface even though the
//! intended target is a sidecar two containers away — the classic payload
//! being a cloud metadata endpoint on `169.254.169.254`.
//!
//! The response would not deserialize into an [`InferResult`], so this is
//! blind SSRF rather than a data-exfiltration primitive. Blind is still
//! enough: it reaches internal-only listeners and it makes Armor the
//! attacker's HTTP client.
//!
//! # What it does and does not block
//!
//! Private ranges are **allowed** — the sidecar is internal, so denying them
//! would deny the only deployment that exists. Loopback is allowed for the
//! same reason (`cargo run` beside a local sidecar). What gets denied is the
//! set of addresses that have no legitimate reading as "our inference pool":
//! link-local (which is where every cloud's metadata service lives),
//! unspecified, multicast, and broadcast.
//!
//! # Rebinding
//!
//! Resolution happens **once**, here, and the validated addresses are pinned
//! into the `reqwest` client. Every later request goes to the address that
//! was checked, so DNS cannot return something benign for the check and
//! something else for the connection.
//!
//! [`InferResult`]: crate::contract::InferResult

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

/// Why an endpoint was refused. Separate from [`InferError`] because these
/// are configuration faults found at startup, not call failures — they should
/// stop a deployment coming up, not degrade one request.
///
/// [`InferError`]: crate::transport::InferError
#[derive(Debug, thiserror::Error)]
pub enum EndpointError {
    #[error("inference endpoint {0:?} is not a valid URL: {1}")]
    Malformed(String, String),
    #[error("inference endpoint {0:?} must use http or https, not {1:?}")]
    UnsupportedScheme(String, String),
    #[error("inference endpoint {0:?} has no host")]
    MissingHost(String),
    #[error("inference endpoint {0:?} did not resolve: {1}")]
    Unresolvable(String, String),
    #[error("inference endpoint {url:?} resolves to {addr}, which is {reason}")]
    Blocked {
        url: String,
        addr: IpAddr,
        reason: &'static str,
    },
}

/// Whether `addr` is a plausible address for an inference pool, or the name
/// of the reason it is not.
///
/// Returns `Err(reason)` rather than a bool so the rejection can say which
/// rule fired — an operator staring at a blocked endpoint needs to know
/// whether they typo'd a host or whether their DNS is lying to them.
pub fn check_addr(addr: IpAddr) -> Result<(), &'static str> {
    match addr {
        IpAddr::V4(v4) => check_v4(v4),
        IpAddr::V6(v6) => {
            // An IPv4-mapped address (`::ffff:169.254.169.254`) is a v4
            // destination wearing a v6 type. Checking only the v6 rules
            // against it is the standard way this guard gets bypassed, so
            // unwrap it first and apply the v4 rules to what is actually
            // being dialled.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return check_v4(mapped);
            }
            check_v6(v6)
        }
    }
}

fn check_v4(addr: Ipv4Addr) -> Result<(), &'static str> {
    // 169.254.0.0/16. The reason this guard exists: every major cloud serves
    // instance credentials from 169.254.169.254.
    if addr.is_link_local() {
        return Err("link-local (the cloud metadata range)");
    }
    if addr.is_unspecified() || addr.octets()[0] == 0 {
        // 0.0.0.0/8 — "this network". Connecting to 0.0.0.0 reaches localhost
        // on Linux, so it is a loopback alias that skips a loopback check.
        return Err("in 0.0.0.0/8, which is not a routable destination");
    }
    if addr.is_multicast() {
        return Err("multicast");
    }
    if addr.is_broadcast() {
        return Err("the broadcast address");
    }
    // Private, loopback, shared (100.64/10) and public all pass: the sidecar
    // is normally on a private network, sometimes on loopback in development,
    // and could legitimately be a hosted pool.
    Ok(())
}

/// AWS's own docs call this address "link-local" and serve IMDS from it on
/// any Nitro instance with IPv6 enabled — but it lives in `fd00::/8`
/// (unique-local), not `fe80::/10`, so the standard v6 link-local check below
/// does not see it. Blocking all of `fd00::/8` to catch it would defeat the
/// "private is allowed" rule this guard exists to preserve for v6 (ULA is the
/// v6 analogue of RFC 1918 space); blocking this one documented address does
/// not.
const AWS_IPV6_METADATA_ADDR: Ipv6Addr = Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254);

fn check_v6(addr: Ipv6Addr) -> Result<(), &'static str> {
    if addr.is_unspecified() {
        return Err("the unspecified address");
    }
    if addr.is_multicast() {
        return Err("multicast");
    }
    if addr == AWS_IPV6_METADATA_ADDR {
        return Err("the AWS IPv6 metadata address (fd00:ec2::254)");
    }
    // fe80::/10 — the v6 link-local range.
    if (addr.segments()[0] & 0xffc0) == 0xfe80 {
        return Err("link-local");
    }
    Ok(())
}

/// Resolve `url`'s host once and return the addresses to pin, or explain why
/// the endpoint is unusable.
///
/// **If any resolved address is denied, the whole endpoint is refused** — not
/// just that address. A host that resolves to a metadata address *at all* is
/// anomalous, and quietly connecting to whichever of its addresses happened
/// to pass would turn a loud misconfiguration into a silent one that depends
/// on resolver ordering.
pub async fn resolve_endpoint(url: &str) -> Result<(reqwest::Url, Vec<SocketAddr>), EndpointError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| EndpointError::Malformed(url.to_string(), e.to_string()))?;

    let scheme = parsed.scheme().to_string();
    if scheme != "http" && scheme != "https" {
        return Err(EndpointError::UnsupportedScheme(url.to_string(), scheme));
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| EndpointError::MissingHost(url.to_string()))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| EndpointError::MissingHost(url.to_string()))?;

    let resolved: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|e| EndpointError::Unresolvable(url.to_string(), e.to_string()))?
        .collect();

    if resolved.is_empty() {
        return Err(EndpointError::Unresolvable(
            url.to_string(),
            "no addresses returned".to_string(),
        ));
    }

    for addr in &resolved {
        if let Err(reason) = check_addr(addr.ip()) {
            return Err(EndpointError::Blocked {
                url: url.to_string(),
                addr: addr.ip(),
                reason,
            });
        }
    }

    Ok((parsed, resolved))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(s: &str) -> IpAddr {
        IpAddr::V4(s.parse().unwrap())
    }

    fn v6(s: &str) -> IpAddr {
        IpAddr::V6(s.parse().unwrap())
    }

    #[test]
    fn cloud_metadata_is_blocked() {
        // The payload this guard exists for.
        assert!(check_addr(v4("169.254.169.254")).is_err());
        assert!(check_addr(v4("169.254.0.1")).is_err());
    }

    #[test]
    fn an_ipv4_mapped_metadata_address_is_blocked_too() {
        // The standard bypass: same destination, v6 type, so a guard that
        // only applies v6 rules to v6 addresses waves it through.
        assert!(check_addr(v6("::ffff:169.254.169.254")).is_err());
        assert!(check_addr(v6("::ffff:0.0.0.0")).is_err());
    }

    #[test]
    fn zero_page_addresses_are_blocked() {
        // 0.0.0.0 reaches localhost on Linux — a loopback alias that would
        // slip past a naive "is it 127.x?" check.
        assert!(check_addr(v4("0.0.0.0")).is_err());
        assert!(check_addr(v4("0.1.2.3")).is_err());
    }

    #[test]
    fn multicast_and_broadcast_are_blocked() {
        assert!(check_addr(v4("224.0.0.1")).is_err());
        assert!(check_addr(v4("255.255.255.255")).is_err());
        assert!(check_addr(v6("ff02::1")).is_err());
    }

    #[test]
    fn ipv6_link_local_is_blocked() {
        assert!(check_addr(v6("fe80::1")).is_err());
        assert!(check_addr(v6("febf::1")).is_err());
    }

    #[test]
    fn private_addresses_are_allowed_because_that_is_where_the_sidecar_lives() {
        assert!(check_addr(v4("10.0.0.5")).is_ok());
        assert!(check_addr(v4("172.16.0.1")).is_ok());
        assert!(check_addr(v4("192.168.1.10")).is_ok());
        // Unique-local v6 — the container-network equivalent.
        assert!(check_addr(v6("fd00::1")).is_ok());
    }

    #[test]
    fn the_aws_ipv6_metadata_address_is_blocked_even_though_it_is_unique_local() {
        // fd00:ec2::254 lives in fd00::/8, the same "allowed as private" range
        // as the test above — this is the one address in that range that is
        // not actually private.
        assert!(check_addr(v6("fd00:ec2::254")).is_err());
        // A sibling address in the same /32 that is NOT the metadata address
        // must stay allowed — this guards against overcorrecting to a
        // blanket fd00:ec2::/32 (or wider) block.
        assert!(check_addr(v6("fd00:ec2::1")).is_ok());
    }

    #[test]
    fn loopback_is_allowed_for_local_development() {
        assert!(check_addr(v4("127.0.0.1")).is_ok());
        assert!(check_addr(v6("::1")).is_ok());
    }

    #[test]
    fn public_addresses_are_allowed_for_a_hosted_pool() {
        assert!(check_addr(v4("93.184.216.34")).is_ok());
    }

    #[tokio::test]
    async fn a_non_http_scheme_is_refused() {
        let err = resolve_endpoint("file:///etc/passwd").await.unwrap_err();
        assert!(matches!(err, EndpointError::UnsupportedScheme(..)));
        let err = resolve_endpoint("gopher://example.com/").await.unwrap_err();
        assert!(matches!(err, EndpointError::UnsupportedScheme(..)));
    }

    #[tokio::test]
    async fn a_malformed_url_is_refused() {
        assert!(matches!(
            resolve_endpoint("not a url").await.unwrap_err(),
            EndpointError::Malformed(..)
        ));
    }

    #[tokio::test]
    async fn a_literal_metadata_endpoint_is_refused_without_dns() {
        let err = resolve_endpoint("http://169.254.169.254/latest/meta-data/")
            .await
            .unwrap_err();
        match err {
            EndpointError::Blocked { addr, reason, .. } => {
                assert_eq!(addr.to_string(), "169.254.169.254");
                assert!(reason.contains("link-local"));
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_loopback_endpoint_resolves_and_pins() {
        let (url, addrs) = resolve_endpoint("http://127.0.0.1:9000").await.unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0].port(), 9000);
    }

    #[tokio::test]
    async fn the_default_port_is_filled_in_from_the_scheme() {
        let (_, addrs) = resolve_endpoint("http://127.0.0.1").await.unwrap();
        assert_eq!(addrs[0].port(), 80);
        let (_, addrs) = resolve_endpoint("https://127.0.0.1").await.unwrap();
        assert_eq!(addrs[0].port(), 443);
    }
}
