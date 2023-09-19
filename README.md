# RepoStats
Query and generate statistics and information from any local git repository. Supports 3 types of queries. More info and demo in main.ipynb, but can be used directly without main.ipynb.

## Query for modifications and stats
Query within a range of commits (inclusive) for a list of modified files/folders (or ones with a modified file somewhere down the directory subtree). Result list contains only files/folders up to the specified depth relative to the specified directory. Various parameters can be set to restrict the definition of a "modified" file, which will only alter the result list and other stats computed based on modified files.

## Commits affecting paths
Get a dictionary of the number of modifications made by each commit to a path or within its subpaths.

## Existential test
Whether a file existed, came into existence, ceased to exist or never existed within the range of commits specified, for all the files that ever existed in the whole repo history.

## Files
./lib.py: RepoStats class with API and main algorithm

./structures.py: implementation of data structures

./repo.py: generator function to process a local git repo using pydriller

./main.ipynb: walkthrough of features and specification