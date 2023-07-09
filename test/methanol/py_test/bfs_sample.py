from collections import deque

class Node:
    """

    ノードの情報を管理

    Attributes:
        index (int): 自分のノード番号
        nears (list): 隣接リスト
        parent (int): 親のノード番号
    """

    def __init__(self, index):
        self.index = index
        self.nears = []
        self.parent = -1

    def __repr__(self):
        return f"(index:{self.index}, nears:{self.nears}, parent:{self.parent})"


# 入力読込
print(input())
N, M = map(int, input().split())
# Nodeインスタンスを作成しnodesに格納
nodes = [Node(i) for i in range(N + 1)]

# 隣接リストを付与
for _ in range(M):
    s, g = map(int, input().split())
    nodes[s].nears.append(g)
    nodes[g].nears.append(s)

# 探索キューを作成
queue = deque([])

# ノード1からBFS開始
queue.append(nodes[1])

# ノード1の親ノードを便宜上0とする
nodes[1].parent = 0

# BFS 開始
while queue:
    # キューから探索先を取り出す
    node = queue.popleft()
    # print(node)←現在地を出力
    # 現在地の隣接リストを取得
    nears = node.nears
    for near in nears:
        # 親ノードが-1なら未探索
        if nodes[near].parent == -1:
            # 未探索ノードをキューに追加
            queue.append(nodes[near])
            # 親ノードを追加
            nodes[near].parent = node.index

# 親ノードを格納
ans = [nodes[i].parent for i in range(2, N + 1)]
# -1が含まれていたらノード1に辿り着けないノードが存在する
if -1 in ans:
    print("No")
else:
    print("Yes")
    for a in ans:
        print(a)

