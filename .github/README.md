# README maintenance

Keep the repository root `README.md` as a concise, English-language guide for ComfyUI users, not a developer manual.

- Explain the plugin's purpose in plain language and show the example workflow image from `examples/` near the top.
- Keep installation, LM Studio setup and the first-run workflow short and actionable. Document only settings and behavior that exist in the current plugin.
- List every registered node using its ComfyUI display name and a one-sentence explanation of its user-facing role. Check `src/__init__.py` when updating the list.
- Summarize the major user-visible changes for each release represented in `CHANGELOG.md`. Keep the full technical history in the changelog; do not infer release or publication dates from commit dates.
- Update the README whenever installation, settings, node names or roles, workflow wiring, outputs, or release highlights change. Remove instructions for features that have been removed.
- Link to `examples/README.md` for detailed workflow guidance and to `CHANGELOG.md` for the complete history. Keep internal architecture, implementation notes and development checks out of the user guide unless users need them to install or operate the plugin.

Before finishing, verify that the example image path exists, node names match the registration, release summaries match the changelog, and all setup instructions match the current code and package files.
