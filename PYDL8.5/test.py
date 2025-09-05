import os, re, collections
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydl85 import DL85Classifier
import pprint

PATH = "datasets/mushroom.txt"

def peek_file(path, n=5):
    print(f"\n--- First {n} lines of {path} ---")
    with open(path) as f:
        for i, line in enumerate(f):
            if i>=n: break
            print(line.rstrip())

def detect_dense(tokens):
    # Dense if tokens are all 0/1
    return all(t in ("0","1") for t in tokens)

def load_any_binary_file(path, label_position="first"):
    """Loads either dense-binary (0/1 per feature) or transaction (item IDs) format."""
    rows = []
    with open(path) as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            if label_position == "first":
                label_tok, data_toks = toks[0], toks[1:]
            elif label_position == "last":
                label_tok, data_toks = toks[-1], toks[:-1]
            else:
                raise ValueError("label_position must be 'first' or 'last'")
            rows.append((label_tok, data_toks))

    if not rows:
        raise ValueError("File is empty.")

    # Decide dense vs transaction from the first data row
    is_dense = detect_dense(rows[0][1])

    if is_dense:
        # Dense: just map 0/1 tokens to ints
        y = np.array([int(lbl) for lbl,_ in rows], dtype=np.int64)
        X = np.array([[int(t) for t in toks] for _, toks in rows], dtype=np.uint8)
        return X, y, "dense"
    else:
        # Transaction: item id list → 0/1 matrix
        labels = []
        transactions = []
        max_id = 0
        for lbl_tok, toks in rows:
            labels.append(int(lbl_tok))
            ids = []
            for t in toks:
                m = re.match(r'^(-?\d+)', t)
                if not m:
                    raise ValueError(f"Cannot parse item token '{t}'")
                ids.append(int(m.group(1)))
            transactions.append(ids)
            if ids:
                max_id = max(max_id, max(ids))
        X = np.zeros((len(transactions), max_id + 1), dtype=np.uint8)
        for i, ids in enumerate(transactions):
            pos = [j for j in ids if 0 <= j <= max_id]
            X[i, pos] = 1
        if X.shape[1] > 0 and X[:,0].sum() == 0:
            X = X[:,1:]
        y = np.asarray(labels, dtype=np.int64)
        return X, y, "transaction"

def dl85_tree_to_dot(tree, feature_names=None):
    node = tree[0] if isinstance(tree, (list, tuple)) and tree and isinstance(tree[0], dict) else tree
    if not isinstance(node, dict):
        raise TypeError("Unsupported tree_ structure. Expected dict (or list of dicts).")

    def pick(keys):
        for k in keys:
            if k in node: return k
        return None

    feature_key = pick(['feat','feature','att','attribute','var','item'])
    class_key   = pick(['class','label','prediction','pred','value'])
    left_key    = pick(['left','l','false','0','left_child'])
    right_key   = pick(['right','r','true','1','right_child'])
    children_key = pick(['children','childs'])

    dot = ["digraph Tree {",
           'node [shape=box, style="filled", color="black", fontname="helvetica"];',
           'edge [fontname="helvetica"];']
    counter = {'i':0}
    def new_id(): counter.__setitem__('i', counter['i']+1); return counter['i']

    def node_label(n):
        if class_key and class_key in n and (not feature_key or feature_key not in n):
            return f"class = {n[class_key]}", True
        if feature_key and feature_key in n:
            idx = n[feature_key]
            name = feature_names[idx] if feature_names is not None and 0 <= idx < len(feature_names) else f"f{idx}"
            return name, False
        return "\\n".join([f"{k}={v}" for k,v in n.items()]), False

    def recurse(n):
        nid = new_id()
        label, is_leaf = node_label(n)
        fill = "#e58139" if is_leaf else "#39e5e5"
        dot.append(f'{nid} [label="{label}", fillcolor="{fill}"];')
        if not is_leaf:
            if children_key and children_key in n and isinstance(n[children_key], (list, tuple)) and len(n[children_key])>=2:
                L, R = n[children_key][0], n[children_key][1]
            else:
                L = n.get(left_key); R = n.get(right_key)
            if L is not None:
                lid = recurse(L); dot.append(f'{nid} -> {lid} [label="0"]')
            if R is not None:
                rid = recurse(R); dot.append(f'{nid} -> {rid} [label="1"]')
        return nid

    recurse(node)
    dot.append("}")
    return "\n".join(dot)

# -------- RUN --------
print("\nPeeking at file:")
peek_file(PATH, 5)

# Try both label positions; pick the one with more varying columns
stats = []
for pos in ["first", "last"]:
    X, y, mode = load_any_binary_file(PATH, label_position=pos)
    n, d = X.shape
    col_sums = X.sum(axis=0)
    varying = ((col_sums > 0) & (col_sums < n))
    stats.append((varying.sum(), pos, mode, X, y))
    print(f"\nLabel position = {pos} | mode = {mode}")
    print(f"Shape: {X.shape} | y classes: {collections.Counter(y)}")
    print(f"Columns: all-zero={(col_sums==0).sum()}, all-one={(col_sums==n).sum()}, varying={int(varying.sum())}")

# choose the parse with max varying features
stats.sort(reverse=True, key=lambda t: t[0])
varying_count, label_pos, mode, X, y = stats[0]
print(f"\n-> Using parse: label={label_pos}, mode={mode}, varying={int(varying_count)}")

if varying_count == 0:
    raise SystemExit("No varying features — check the file format/contents.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
clf = DL85Classifier(max_depth=10, min_sup=5)
clf.fit(X_train, y_train)
print("\nTrain acc:", clf.accuracy_)
print("Test  acc:", accuracy_score(y_test, clf.predict(X_test)))
pprint.pp(clf.tree_)

dot_str = dl85_tree_to_dot(clf.tree_, feature_names=[f"f{i}" for i in range(X.shape[1])])
src = graphviz.Source(dot_str)
src.render("dl85_tree", format="png", cleanup=True)
src.view()
print("\nWrote dl85_tree.png")
