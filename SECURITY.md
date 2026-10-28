# Security and Data Privacy

## Reporting a concern

Do not post sensitive information, credentials, private MRI data, participant identifiers, or exploitable details in a public issue. Contact the repository maintainers privately using the contact channel agreed by the project team. Add the private reporting address or institutional contact here once the maintainers establish one.

## Medical data

- Do not commit MRI scans, segmentation masks, participant IDs, clinical records, or identity mappings.
- Keep local datasets in the ignored `data/` directories or approved protected storage.
- Do not upload private data to public issue trackers, pull requests, CI logs, or public cloud services without explicit authorization.
- Check data licenses, consent, institutional approvals, and access requirements.
- Share anonymized aggregate summaries rather than subject-level details whenever possible.

## Secrets and local configuration

- Never commit credentials, tokens, private keys, or `.env` files containing secrets.
- `.env.example` must contain placeholders only.
- If a secret is committed accidentally, revoke/rotate it immediately and contact maintainers. Removing it from the latest commit alone may not remove it from Git history.

## Software and dependencies

Use trusted package sources and official installation guidance. Review dependency changes and keep environments isolated. Report suspected dependency or supply-chain issues privately to maintainers.

## Scope

This is a research codebase, not a clinical medical device. It does not provide a guarantee of diagnostic accuracy or clinical safety. Any clinical use would require separate validation, governance, and regulatory review.
