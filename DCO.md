# Developer Certificate of Origin

Contributions to this repository require a **sign-off**, not a signed
agreement. This project is Apache-2.0, full stop — no split licensing, no
relicensing plans — so there's nothing here that needs a contributor to
grant rights beyond what Apache-2.0 already grants everyone. A sign-off is
just your attestation that you wrote the contribution, or otherwise have
the right to submit it under this project's license.

## What you're attesting

By adding a `Signed-off-by` line to a commit, you certify the
[Developer Certificate of Origin, version 1.1](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## How to sign off

Add `-s` to your commit:

```
git commit -s -m "Fix redaction span off-by-one"
```

This appends a trailer to the commit message using your git `user.name` /
`user.email`:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Sign off under a name you are willing to stand behind — the DCO is an
attestation, and it should be traceable to you.

The address does not have to be one you read mail at. A GitHub noreply
address (`ID+username@users.noreply.github.com`, found under **Settings →
Emails**) is fine and is what we suggest if you would rather not publish a
personal address: git history is permanent and public, and this project asks
nobody to expose an inbox as the price of contributing. GitHub still ties
those commits to your account, which is the identity link the DCO cares
about.

**Forgot to sign off?** Amend the last commit:

```
git commit --amend -s --no-edit
```

For multiple commits in a branch, `git rebase --signoff main` adds the
trailer to every commit since it diverged from `main`.

## Enforcement

`.github/workflows/dco.yml` checks every commit in a pull request for a
`Signed-off-by` trailer and fails the check if one is missing — see that
file for exactly what it checks. Fix it with the amend/rebase commands
above and push again; the check re-runs automatically.
