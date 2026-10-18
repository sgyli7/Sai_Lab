# V5 aliases-aware independent audit V2

Source SHA `81c20e52fe258ace20385d5e7d643a003558e32174d44f8be1d8cfdee0b2f782` (163lines); V1 `1690d532d6a116c3d0694d121fbc902d044cff4a80f06047f0e28f3c3fc65ff0` remainsunchanged. Five newtinyCPUtests passed. No actualarchive/130GBsource scan orGPUoperation.

```bash
python -m experiments.pi05_g1_mobile.audit_archive_v5_aliases_v2 --archive /ABS/ROOT_APPROVED_ARCHIVE --output /ABS/FRESH_REPORT_OUTSIDE_ARCHIVE.json
```

Report schema `pi05_g1_mobile_archive_independent_audit_aliases_v2`, statuspassed/failed. Archivedsourcefiles neednotexist. Allaliastranslation islexical fromexplicitarchivedplan declarations; everymapping pairs exactlywith sources.path/destination/original_alias_path. Duplicate/overlapping/cross-ambiguous roots ordestinations andunsafe/noncanonical paths arerejected.

Requiredclosure lookupfirsttries ordinary source_path/source_resolved_path exactmatches. Onlyonmiss doesit usePathcomponent aliasprefix toresolved/suffix andrequire logicalpath preciselyretained/destination/suffix. Similar stringprefixes cannotmap. ExistingV1byte/header/closure/finalfield auditlogic remainsunchanged.

Diff `v1_to_aliases_v2.diff`; testslog/receipt included. Actualfullarchiveaudit remainsRoot authorizationdecision.
