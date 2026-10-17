#!/usr/bin/env bash
# Verify every file in code/ is byte-identical to the as-run file it was copied from (see MANIFEST.tsv).
cd "$(dirname "$0")/.." || exit 1
fail=0
while IFS=$'\t' read -r copy sha src; do
  [ "$copy" = reading_copy ] && continue
  a=$(sha256sum "code/$copy" | cut -d' ' -f1); b=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$a" != "$sha" ] || [ "$b" != "$sha" ]; then echo "MISMATCH code/$copy"; fail=1; fi
done < code/MANIFEST.tsv
[ $fail = 0 ] && echo "all $(($(wc -l < code/MANIFEST.tsv) - 1)) reading copies match their as-run files"
exit $fail
