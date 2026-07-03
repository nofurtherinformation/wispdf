# Changesets

This directory is managed by [Changesets](https://github.com/changesets/changesets).

## Release flow

1. After merging a feature or fix PR, run `npx changeset add` to describe the change
   and bump type (patch / minor / major).
2. When ready to release, run `npx changeset version` to consume all pending changesets,
   update `package.json` version, and update `CHANGELOG.md`.
3. Build and publish: `npm run build && npm publish`.

The `.changeset/config.json` is the minimal configuration. Install
`@changesets/cli` as a devDep if you want to use the CLI directly:

```
npm install -D @changesets/cli
```
