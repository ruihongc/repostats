# module to read local git repo
# can use gitpython or pydriller
from pydriller import Repository
from tqdm import tqdm
from structures import DotDict
import subprocess

def metrics(file):
    # variance squared average distance between edits minus average distance between edits squared
    lines = (
        file.source_code.count("\n")
        if file.source_code
        else 0
    ) + (
        file.source_code_before.count("\n")
        if file.source_code_before
        else 0
    
    )
    lines2 = 0
    lines_t = 0
    lines_changed = 0
    count = 0
    prev = [0, 0]

    for line in file.diff.split("\n"):
        if line.startswith("+") or line.startswith("-"):
            # in only one file
            # assuming changes are not repeated in multiple hunks AND hunks are sorted
            lines2 += count * count
            lines_t += count
            count = 1
            lines_changed += 1
        elif line.startswith("@@"):
            token = [i.split(",") for i in line.split(" ")[1:3]]
            for i in range(2):
                v = token[i]
                v0 = int(v[0][1:])
                count += v0 - prev[i]
                prev[i] = v0
                if len(v) > 1:
                    prev[i] += int(v[1])
                else:
                    prev[i] += 1
        elif line == r"\ No newline at end of file":
            pass
        else:
            if count: # after the first diff
                count += 2 # in both files
    if lines_changed > 0:
        variance = (lines2 - (lines_t * lines_t / lines_changed)) / lines_changed
    else:
        variance = 0
    return DotDict(
        lines = lines,
        lines_changed = lines_changed,
        variance = variance,
    )

def read_commits(path, from_commit, to_commit):
    repo = Repository(path, from_commit=from_commit, to_commit=to_commit)
    total = 1 + int(subprocess.check_output(f"cd {path} && git rev-list {from_commit}...{to_commit} | wc -l", shell=True))
    for i in tqdm(repo.traverse_commits(), total = total, desc = "Processing commits"):
        if i.merge: # ignore merge commits
            continue
        yield DotDict(
            hash = i.hash,
            date = i.committer_date, # doesnt handle timezone
        )
        for j in i.modified_files:
            m = metrics(j)
            yield DotDict(
                old_path = j.old_path,
                new_path = j.new_path,
                change_type = j.change_type,
                lines = m.lines, # combined total of lines originally and lines in resultant file
                lines_changed = m.lines_changed, # combined total of added lines and removed lines
                percentage = (m.lines_changed / m.lines) if m.lines else .0,
                variance = m.variance,
            )
