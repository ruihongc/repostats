import pickle
from types import MethodType
from structures import DotDict, SparseList2D, Tree
from tqdm import tqdm
from repo import read_commits
import os
import numpy as np

def sanitise_path(path):
    return path.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'").replace("/", "']['")

def check_range(val, val_min, val_max):
    return ((val >= val_min) and ((val_max < val_min) or (val <= val_max)))

class RepoStats:
    def __init__(self, *, path, from_commit, to_commit):
        path = os.path.expanduser(path)
        self.info = DotDict(
            path = path,
            from_commit = from_commit,
            to_commit = to_commit,
        )

        # read paths in directory
        # the following block can be commented out to save time
        # if unmodified files are not needed
        # but this hasn't been the bottleneck so far
        # so leaving it here for now
        visited = {}
        path_len = len(path) + 1
        for root, _, files in tqdm(os.walk(path,  topdown=True), desc = "Walking directory"):
            for name in files:
                fullname = os.path.join(root, name)[path_len:]
                if fullname not in visited:
                    visited[fullname] = True

        self.sparse = DotDict(
            changes = SparseList2D(),
            percentage = SparseList2D(),
            percentage2 = SparseList2D(),
            variance = SparseList2D(),
            variance2 = SparseList2D(),
            lines = SparseList2D(),
            lines_changed = SparseList2D(),
            commits = SparseList2D(),
            exist = SparseList2D(),
        )
        self.commit_index = {}
        self.commit_hash = []
        self.commit_date = {}
        self.commit_count = 0
        temp_commits = {}

        # process commits
        gen = read_commits(path, from_commit, to_commit)
        for stat in gen:
            if stat.lines:
                v = (stat.old_path, stat.new_path)
                for m in range(2):
                    if v[m] != None:
                        if v[m] not in visited:
                            visited[v[m]] = True
                        if (m == 1) and (stat.change_type in (1, 2)):
                            # add or copy
                            self.sparse.exist.set(v[m], self.commit_count, 1)
                        elif (m == 0) and (stat.change_type == 4):
                            # delete
                            self.sparse.exist.set(v[m], self.commit_count, -1)
                        elif stat.change_type == 3:
                            # rename
                            if m == 1:
                                # renamed to file
                                self.sparse.exist.set(v[m], self.commit_count, 1)
                            else:
                                # renamed from file
                                self.sparse.exist.set(v[m], self.commit_count, -1)
                        self.sparse.changes.set(v[m], self.commit_count, 1)
                        self.sparse.percentage.set(v[m], self.commit_count, stat.percentage)
                        self.sparse.percentage2.set(v[m], self.commit_count, stat.percentage*stat.percentage)
                        self.sparse.variance.set(v[m], self.commit_count, stat.variance)
                        self.sparse.variance2.set(v[m], self.commit_count, stat.variance*stat.variance)
                        self.sparse.lines.set(v[m], self.commit_count, stat.lines)
                        self.sparse.lines_changed.set(v[m], self.commit_count, stat.lines_changed)
                        if self.commit_count in temp_commits:
                            temp_commits[v[m]].append(self.commit_count)
                        else:
                            temp_commits[v[m]] = [self.commit_count]
            else:
                self.commit_index[stat.hash] = self.commit_count
                self.commit_hash.append(stat.hash)
                self.commit_date[stat.hash] = stat.date
                self.commit_count += 1

        # create all files at the start
        # process all paths and add corresponding nodes to path_tree
        self.original = {}
        self.path_tree = Tree()
        for i in tqdm(visited, desc = "Creating files and building tree"):
            exec(f"self.path_tree['{sanitise_path(i)}']") # somewhat dirty
            if self.sparse.exist.first(i) == 1:
                self.original[i] = False
            else:
                self.original[i] = True

        # DFS path_tree to calculate start and end ranges of each node
        # when representing the tree with a flat array using the Euler
        # tour technique. Also returns an ordered array containing the
        # corresponding paths to the index of each leaf node in the tree.
        self.start_range = {}
        self.end_range = {}
        self.paths = []
        self.file_count = len(visited)
        visited = {}
        stack = []
        stack.append("") # path
        count = 0
        with tqdm(total = self.file_count * 2, desc = "Traversing tree") as pbar:
            while stack:
                path = stack.pop()
                if path in visited:
                    # second visit wrap up
                    if self.start_range[path] == count: # leaf node
                        self.paths.append(path)
                        if path in temp_commits:
                            for c in temp_commits[path]:
                                self.sparse.commits.set(c, count, 1)
                        count += 1
                    self.end_range[path] = count
                    pbar.update(1)
                else:
                    # first visit
                    stack.append(path) # add ownself back to top of stack
                    self.start_range[path] = count
                    if path:
                        for i in eval(f"self.path_tree['{sanitise_path(path)}']"):
                            stack.append(path + "/" + i)
                    else:
                        for i in self.path_tree:
                            stack.append(i)
                    visited[path] = True
                    pbar.update(1)

        # calculate lengths for all sparse lists
        for i in tqdm(self.sparse, desc = "Calculating lengths"):
            self.sparse[i].calc_lengths()

    def modifications(self, params):
        """dictionary of how many modifications are made by each commit to a path or within its subpaths"""
        (
            commit1,
            commit2,
            subfolder,
        ) = (
            # required params
            self.commit_index[params.commit1],
            self.commit_index[params.commit2] + 1,
            # optional params
            params.subfolder or "",
        )
        return {
            self.commit_hash[i]: self.sparse.commits.sum(i, self.start_range[subfolder], self.end_range[subfolder])
            for i in tqdm(range(commit1, commit2), desc = "Reducing 2D array")
        }

    def existence(self, params):
        """dictionary of {file path: existence}"""
        (
            commit1,
            commit2,
            subfolder,
        ) = (
            # required params
            self.commit_index[params.commit1],
            self.commit_index[params.commit2] + 1,
            # optional params
            params.subfolder or "",
        )
        existence_changes = [
            "ceased to exist", # -1
            None, # 0
            "came into existence", # 1
        ]
        existences = [
            "never existed", # -1
            "existed", # 1
        ]
        return {
            self.paths[i]: (
                existence_changes[self.sparse.exist.sum(self.paths[i], commit1, commit2) + 1] or
                existences[((self.sparse.exist.before(self.paths[i], commit1) or self.original[self.paths[i]]) + 1) // 2]
            )
            for i in tqdm(range(self.start_range[subfolder], self.end_range[subfolder]), desc = "Reducing 2D array")
        }

    def query(self, params):
        """Query for repo stats within a range of commits and various optional parameters"""
        (
            commit1,
            commit2,
            subfolder,
            depth,
            min_changes,
            max_changes,
            avg_min,
            avg_variance_min,
            variance_min,
            variance_variance_min,
            freq_avg_min,
            avg_max,
            avg_variance_max,
            variance_max,
            variance_variance_max,
            freq_avg_max,
        ) = (
            # required params
            self.commit_index[params.commit1],
            self.commit_index[params.commit2] + 1,
            # optional params
            params.subfolder or "",
            params.depth or -1,
            params.min_changes or 1,
            params.max_changes or -1,
            params.avg_min or 0,
            params.avg_variance_min or 0,
            params.variance_min or 0,
            params.variance_variance_min or 0,
            params.freq_avg_min or 0,
            params.avg_max or -1,
            params.avg_variance_max or -1,
            params.variance_max or -1,
            params.variance_variance_max or -1,
            params.freq_avg_max or -1,
        )

        file_count = 0
        change_count = 0
        avg = {} # what percentage has each file changed (average)
        avg_variance = {} # what percentage has each file changed (variance of average)
        variance = {} # how evenly spread are the lines changed in each file across the commits (average)
        variance_variance = {} # how evenly spread are the lines changed in each file across the commits (variance of average)
        freq_avg = {} # frequency of change of path and each subpath (average)
        duration = (self.commit_date[params.commit2] - self.commit_date[params.commit1]).total_seconds()

        total = 0 # average percentage change across all modified files (unweighted)
        total_weighted = 0 # average percentage change across all modified files (weighted by the length of each file or hence average percentage of lines changed across all modified files)
        total_length = 0

        # reduce 2D arrays
        flat = np.empty(self.end_range[subfolder] - self.start_range[subfolder], dtype="int64")
        flat_changes = np.empty(self.end_range[subfolder] - self.start_range[subfolder], dtype="int64")
        for i in tqdm(range(self.start_range[subfolder], self.end_range[subfolder]), desc = "Reducing 2D array"):
            path = self.paths[i]
            n = self.sparse.changes.sum(path, commit1, commit2) # number of changes

            # calculate stats
            if n:
                calc_stats = lambda e, e2: (
                    ex := e.sum(path, commit1, commit2) / n,
                    (e2.sum(path, commit1, commit2) / n) - (ex*ex),
                )
                avg[path], avg_variance[path] = calc_stats(self.sparse.percentage, self.sparse.percentage2)
                variance[path], variance_variance[path] = calc_stats(self.sparse.variance, self.sparse.variance2)
                freq_avg[path] = duration / n
                # how to add up all days in between minus all intersections then divide by total commits
            else:
                avg[path], avg_variance[path], variance[path], variance_variance[path], freq_avg[path] = (0, 0, 0, 0, float('inf'))

            # test if file satisfies param requirements
            if (
                check_range(n, min_changes, max_changes) and
                check_range(avg[path], avg_min, avg_max) and
                check_range(avg_variance[path], avg_variance_min, avg_variance_max) and
                check_range(variance[path], variance_min, variance_max) and
                check_range(variance_variance[path], variance_variance_min, variance_variance_max) and
                check_range(freq_avg[path], freq_avg_min, freq_avg_max)
            ):
                flat[i - self.start_range[subfolder]] = 1
                file_count += 1
                change_count += n
                total += avg[path]
                total_weighted += self.sparse.lines_changed.sum(path, commit1, commit2)
                total_length += self.sparse.lines.sum(path, commit1, commit2)
            else:
                flat[i - self.start_range[subfolder]] = 0
            flat_changes[i - self.start_range[subfolder]] = n

        # calculate prefix sum and totals
        prefix = flat.cumsum()
        prefix_changes = flat_changes.cumsum()
        if file_count:
            total /= file_count
        if total_length:
            total_weighted /= total_length
        
        # traverse tree and return info
        stack = []
        results = []
        stack.append((subfolder, depth)) # path
        while stack:
            path, d = stack.pop()
            if self.start_range[path] - 1 >= 0:
                s = prefix[self.end_range[path] - 1] - prefix[self.start_range[path] - 1]
                s2 = prefix_changes[self.end_range[path] - 1] - prefix_changes[self.start_range[path] - 1]
            else:
                s = prefix[self.end_range[path] - 1]
                s2 = prefix_changes[self.end_range[path] - 1]
            if s > 0:
                # requirements satisfied hence appending to results
                results.append(path)
            if s2:
                freq_avg[path] = duration / s2
            else:
                freq_avg[path] = float('inf')
            if (depth < 0) or (d > 0):
                # branch to subtrees
                if path:
                    for i in eval(f"self.path_tree['{sanitise_path(path)}']"):
                        stack.append((path + "/" + i, d - 1))
                else:
                    for i in self.path_tree:
                        stack.append((i, d - 1))
        return DotDict(
            results = results,
            files = self.end_range[subfolder] - self.start_range[subfolder], # total number of files
            files_mod = file_count, # number of files considered modified based on query params
            change_count = change_count, # aggregate number of modifications across all "modified" files
            avg = avg, # what percentage has each file changed (average)
            avg_variance = avg_variance, # what percentage has each file changed (variance of average)
            variance = variance, # how evenly spread are the lines changed in each file across the commits (average)
            variance_variance = variance_variance, # how evenly spread are the lines changed in each file across the commits (variance of average)
            total = total, # average percentage change across all "modified" files (unweighted)
            total_weighted = total_weighted, # average percentage change across all "modified" files (weighted by the length of each file or hence average percentage of lines changed across all "modified" files)
            freq_avg = freq_avg, # frequency of change of path and each subpath (average number of days between changes)
        )

    def save(self, filename = None):
        if not filename:
            filename = f"{self.info.path}_{self.info.from_commit}_{self.info.to_commit}.pkl"
        filename = os.path.expanduser(filename)
        with open(filename, "wb") as store:
            pickle.dump(self, store)
            print(filename)

    def update_methods(self, obj):
        """to update the methods of an object loaded from pickle"""
        self.update_methods = MethodType(obj.update_methods, self)
        self.save = MethodType(obj.save, self)
        self.query = MethodType(obj.query, self)
        self.modifications = MethodType(obj.modifications, self)

    @staticmethod
    def load(filename):
        filename = os.path.expanduser(filename)
        with open(filename, 'rb') as store:
            return pickle.load(store)
