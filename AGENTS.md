# Agent instructions

This repo is public. It must not contain personal data, in files or in history.

## Before staging or committing

Check every change for personal references:

- real names, nicknames, usernames, or user IDs
- server, channel, or group names
- message text from real conversations
- tokens, keys, or other credentials
- email addresses, phone numbers, street addresses
- local paths, hostnames, or machine names

Examples use made-up English names (`bob`, `alice`, `carol`). Exports, datasets,
aliases, profiles, bot tokens and models stay out of the repo.

On a hit, leave it unstaged and report the file and line to the user. Check
commit messages and PR descriptions the same way.

## Tests

```sh
pip install -e .
python -m unittest discover -s tests
```
