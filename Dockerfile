# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder
WORKDIR /build

# Cache dependency compilation separately from source changes.
#
# Every workspace member in the root Cargo.toml needs its manifest copied and
# a stub source file here, even ones `-p armor-api` doesn't depend on: cargo
# loads the manifest of every member before it resolves anything, so a
# missing one fails the build outright ("failed to load manifest for
# workspace member"). Adding a crate to `[workspace] members` means adding it
# to the three lists below.
COPY Cargo.toml Cargo.lock ./
COPY crates/core/Cargo.toml crates/core/Cargo.toml
COPY crates/api/Cargo.toml crates/api/Cargo.toml
COPY crates/storage/Cargo.toml crates/storage/Cargo.toml
COPY crates/inference-client/Cargo.toml crates/inference-client/Cargo.toml
RUN mkdir -p crates/core/src crates/api/src crates/storage/src crates/inference-client/src \
    && echo "fn main() {}" > crates/api/src/main.rs \
    && echo "" > crates/core/src/lib.rs \
    && echo "" > crates/storage/src/lib.rs \
    && echo "" > crates/inference-client/src/lib.rs \
    && cargo build --release -p armor-api \
    && rm -rf crates/core/src crates/api/src crates/storage/src crates/inference-client/src

COPY migrations ./migrations
COPY crates ./crates
RUN touch crates/core/src/lib.rs crates/api/src/main.rs crates/storage/src/lib.rs \
        crates/inference-client/src/lib.rs \
    && cargo build --release -p armor-api

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /app --shell /usr/sbin/nologin armor
WORKDIR /app

COPY --from=builder /build/target/release/armor-api /usr/local/bin/armor-api
# rules/*.yaml are compiled into the binary above; only policies.yaml is
# read at runtime (ARMOR_POLICY_PATH).
COPY config ./config

USER armor
ENV ARMOR_BIND_ADDR=0.0.0.0:8100
ENV ARMOR_POLICY_PATH=/app/config/policies.yaml
EXPOSE 8100

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8100/healthz || exit 1

ENTRYPOINT ["armor-api"]
