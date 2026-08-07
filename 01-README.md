# Stage 1: PDF Collection

This stage is not a script — it's the source corpus. Populate it with the
PDFs you want to build a knowledge base from (textbooks, handbooks,
technical references, codes/standards, or any other domain literature),
organized however is convenient for you (flat folder, or one subfolder per
source category — Stage 2 scans recursively either way).

There is no code here by design: corpora are project-specific and usually
too large or too rights-restricted to redistribute in a repository. If you
want the folder to be self-documenting, add a `corpus_manifest.csv` (or
similar) listing title, edition, and source domain for each PDF you used.

## Output
A folder of source PDFs, ready to be passed as `--pdf-root` to Stage 2.
