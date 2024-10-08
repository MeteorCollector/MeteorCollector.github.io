---
layout: post
title:  "机试板子备份"
date:   2024-05-28 15:02:00 +0800
categories: posts
tag: oi
---

## 写在前面

正在备战夏令营机试。我本来就是上机苦手，南哪对机试的要求又这么高，唉......现在在刷洛谷，肯定是刷不完了，刷不完就看看 [oi wiki](https://oi-wiki.org/) 吧。

这里主要囤积一些符合自己码风的板子和 stl 的使用参考。

## 数据类型

### 二分

```c++
// first location
// but seems c++ has bisearch stl?

int bi(int l, int r, int key, int arr[])
{
	// cout << "(" << l << ", " << r << ")" << endl;
	if (l >= r) { return -1; }
	int mid = (l + r + 1) / 2;
	int val = arr[mid];
	if (val == key && arr[mid - 1] != key) { return mid; }
	else if (val >= key) { return bi(l, mid - 1, key, arr); }
	else { return bi(mid, r, key, arr); }
}
```

### 上界/下界

```c++
typedef long long ll;
using namespace std;

//P5076

int vector_upper_lower_bound()
{
	int n;
	cin >> n;
	vector<int> arr;

	arr.push_back(-2147483647);
	arr.push_back(2147483647);

	while (n--)
	{
		int op, x;
		scanf("%d%d", &op, &x);
		if (op == 1)
		{
			printf("%d\n", (lower_bound(arr.begin(), arr.end(), x) - arr.begin()));
		}
		else if (op == 2)
		{
			printf("%d\n", arr[x]);
		}
		else if (op == 3)
		{
			printf("%d\n", *(--lower_bound(arr.begin(), arr.end(), x)));
		}
		else if (op == 4)
		{
			printf("%d\n", *upper_bound(arr.begin(), arr.end(), x));
		}
		else if (op == 5)
		{
			auto it = lower_bound(arr.begin(), arr.end(), x);
			arr.insert(it, x);
		}
	}
	return 0;
}


int multiset_upper_lower_bound()
{
	int n;
	cin >> n;

	multiset<int> arr;

	arr.insert(-2147483647);
	arr.insert(2147483647);

	while (n--)
	{
		int op, x;
		scanf("%d%d", &op, &x);
		if (op == 1)
		{
			int target = *arr.lower_bound(x);
			int i = 0;
			auto it = arr.begin();
			while (*it != target) { it++; i++; }
			printf("%d\n", i);
		}
		else if (op == 2)
		{
			auto it = arr.begin();
			for (int i = 0; i < x; i++) { it++; }
			printf("%d\n", *it);
		}
		else if (op == 3)
		{
			printf("%d\n", *(--arr.lower_bound(x)));
		}
		else if (op == 4)
		{
			printf("%d\n", *arr.upper_bound(x));
		}
		else if (op == 5)
		{
			arr.insert(x);
		}
	}
	return 0;
}
```

### P1725 琪露诺 单调队列

```cpp
typedef long long ll;
using namespace std;
const int maxn = 2e5 + 1000;
ll a[maxn], dp[maxn];

class node
{
public:
	ll val; int id;
	node(int a = 0, ll b = 0) { val = b; id = a; }
};

bool operator< (node a, node b)
{
	return a.val < b.val; // 大根堆
}

priority_queue<node> que;

int main()
{
	int N, L, R;
	cin >> N >> L >> R;
	for (int i = 0; i <= N; i++)
	{
		cin >> a[i];
		dp[i] = - LLONG_MIN / 2;
	}
	dp[0] = a[0];

	for (int i = L; i <= N; i++)
	{
		que.push(node(i - L, dp[i - L]));
		while (que.top().id < i - R) { que.pop(); }
		ll maxval = que.top().val;
		dp[i] = a[i] + maxval;
	}

	ll ans = LLONG_MIN / 2;
	for (int i = N - R + 1; i <= N; i++)
	{
		ans = max(dp[i], ans);
	}

	cout << ans;
}
```

### kth_element (qsort)

```c++
int kth(int l, int r, int k, int arr[])
{
	int i = l, j = r, mid = arr[(l + r) / 2];
	do
	{
		while (arr[j] > mid) j--;
		while (arr[i] < mid) i++;
		if (i <= j)
		{
			swap(arr[i], arr[j]);
			i++;
			j--;
		}
	} while (i <= j);

	if (k <= j) return kth(l, j, k, arr);
	else if (i <= k) return kth(i, r, k, arr);
	else
	{
		return mid;
	}
}
```

### 并查集

```c++
// P1551

const int maxn = 1e5;

int pr[maxn] = { 0 };
int rk[maxn] = { 0 };

void init_pr(int n)
{
	for (int i = 1; i <= n; i++)
	{
		pr[i] = i;
	}
}

int __findp(int a)
{
	return pr[a] == a ? a : pr[a] = __findp(pr[a]);
}

void __unionp(int a, int b)
{
	int fa = __findp(pr[a]);
	int fb = __findp(pr[b]);
	if (fa == fb) { return; }
	else { pr[fb] = fa; }
}

void __union_rank(int x, int y) {
	int rx = __findp(x);
	int ry = __findp(y);
	if (rx != ry) {
		if (rk[rx] < rk[ry])   pr[rx] = ry;
		else  if (rk[ry] < rk[rx])  pr[ry] = rx;
		else pr[rx] = ry, rk[ry]++;
	}
	return;
}
```

### 单调栈：柱状图中最大的矩形（每个位置找到左右最近的高度小于它的柱子）

[leetcode链接](https://leetcode.cn/problems/largest-rectangle-in-histogram/solutions/266844/zhu-zhuang-tu-zhong-zui-da-de-ju-xing-by-leetcode-/)

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        vector<int> left(n), right(n);
        
        stack<int> mono_stack;
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.empty() ? -1 : mono_stack.top());
            mono_stack.push(i);
        }

        mono_stack = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.empty() && heights[mono_stack.top()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.empty() ? n : mono_stack.top());
            mono_stack.push(i);
        }
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }
};
```



## 数论

### 费马小定理

当 $p$ 是质数时，$a^{p-1} \equiv 1\quad (\text{mod }p)$，$a \times a^{p-2} \equiv 1\quad (\text{mod }p)$，因此 $a$ 在 $\text{mod }p$ 意义下乘法逆元为 $a^{p-2}$。常结合快速幂求解。

### 快速幂

```
int quickPower(int a, int b)
{
	int ans = 1, base = a;
	while (b)
    {
		if(b & 1)
			ans *= base;
        base *= base;
		b >>= 1;
	}
	return ans;
}
```

### 线性筛质数

```c++
#include <vector>
using namespace std;

// 线性筛，oiwiki https://oi-wiki.org/math/number-theory/sieve/
const int N = 1e6;


vector<int> pri;
bool not_prime[N];

void pre(int n) {
    for (int i = 2; i <= n; ++i) {
        if (!not_prime[i]) {
            pri.push_back(i);
        }
        for (int pri_j : pri) {
            if (i * pri_j > n) break;
            not_prime[i * pri_j] = true;
            if (i % pri_j == 0) {
                // i % pri_j == 0
                // 换言之，i 之前被 pri_j 筛过了
                // 由于 pri 里面质数是从小到大的，所以 i 乘上其他的质数的结果一定会被
                // pri_j 的倍数筛掉，就不需要在这里先筛一次，所以这里直接 break
                // 掉就好了
                break;
            }
        }
    }
}


// 欧拉筛另一个写法，cnt是目前找出的质数的数量
void euler_pre(int n, int B, int p[], int vis[], int cnt) {
    for (int i = 2; i <= B; i++)
    {
        if (!vis[i]) p[++cnt] = i;
        for (int j = 1; j <= cnt && p[j] * i <= B; j++)
        {
            vis[p[j] * i] = 1;
            if (!i % p[j]) break;
        }
    }
}
```



## 字符串

### KMP

```c++
const int maxn = 2e6;
string s1;
string s2;
int kmp[maxn] = { 0 };

int main()
{
	cin >> s1 >> s2;
	int len1 = (int)s1.length();
	int len2 = (int)s2.length();

	kmp[0] = kmp[1] = 0;
	int k = 0;
	// k 代表当前匹配到了第几位

	// 处理 kmp 数组：自己匹配自己
	for (int i = 1; i < len2; i++)
	{
		while (k && (s2[i] != s2[k]))
		{
			k = kmp[k];
		}
		if (s2[i] == s2[k])
		{
			k++;
			kmp[i + 1] = k;
		}
		else
		{
			kmp[i + 1] = 0;
		}
	}

	k = 0;

	// 真正的匹配
	for (int i = 0; i < len1; i++)
	{
		while (k && (s1[i] != s2[k]))
		{
			k = kmp[k];
		}
		if (s1[i] == s2[k])
		{
			k++;
		}
		if (k == len2) { cout << i - len2 + 2 << endl; }
	}

	for (int i = 1; i <= len2; i++)
	{
		cout << kmp[i] << " ";
	}

	return 0;
}
```

进阶版的 BM 算法可以看这里。不过有一说一，真的会考到这种程度吗？

[不用找了，学习BM算法，这篇就够了（思路+详注代码）_bm学习-CSDN博客](https://blog.csdn.net/DBC_121/article/details/105569440)

实在不行还可以用find...

[C++ string中的find()函数 总结-CSDN博客](https://blog.csdn.net/youtiankeng/article/details/109066118)



## 图论

### Prim

```cpp
void Prim() {
  memset(dis, 0x3f, sizeof(dis));
  dis[1] = 0;
  q.push({1, 0});
  while (!q.empty()) {
    if (cnt >= n) break;
    int u = q.top().u, d = q.top().d;
    q.pop();
    if (vis[u]) continue;
    vis[u] = 1;
    ++cnt;
    res += d;
    for (int i = h[u]; i; i = e[i].x) {
      int v = e[i].v, w = e[i].w;
      if (w < dis[v]) {
        dis[v] = w, q.push({v, w});
      }
    }
  }
}
```

### Tarjan 缩点

[P3387 【模板】缩点 - 洛谷 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/solution/P3387)

### Tarjan 判割点

```c++
const int maxn = 3e4;

int N, M;
vector<int> edge[maxn];
int dfncnt = 0;
int dfn[maxn], low[maxn];
queue<int> cut;
bool iscut[maxn] = { 0 };

void tarjan(int src, int root_id)
{
	dfncnt++;
	dfn[src] = low[src] = dfncnt;
	int child_cnt = 0;
	for (int dst : edge[src])
	{
		if (!dfn[dst])
		{
			tarjan(dst, root_id);
			low[src] = min(low[src], low[dst]);
			if (low[dst] >= dfn[src] && src != root_id)
			{
				iscut[src] = true;
			}
			if (src == root_id) { child_cnt++; }
		}
		else
		{
			low[src] = min(low[src], dfn[dst]);
		}
	}
	// if is root, dfn[root] is minimum value, always satisfy low[dst] >= dfn[src].
	if (child_cnt > 1 && src == root_id)
	{ 
		iscut[src] = true;
	}
}

int main()
{
	cin >> N >> M;
	for (int i = 1; i <= M; i++)
	{
		int u, v;
		cin >> u >> v;
		edge[u].push_back(v);
		edge[v].push_back(u);
	}
	
	for (int i = 1; i <= N; i++)
	{
		if (!dfn[i]) { tarjan(i, i); }
	}
	
	int ans = 0;
	for (int i = 1; i <= N; i++)
	{
		if (iscut[i]) ans++;
	}
	cout << ans << endl;
	for (int i = 1; i <= N; i++)
	{
		if (iscut[i]) cout << i << " ";
	}
	
	return 0;
}
```

### Kosaraju

```c++
// B3609
void dfs1(int u)
{
	if (vis[u]) { return; }
	vis[u] = true;
	
	for (int v : edg[u])
	{
		dfs1(v);
	}

	order[orderid] = u; orderid++;
}

void dfs2(int u)
{
	if (vis[u]) { return; }
	vis[u] = true;
	st.push(u);
	for (int v : rev[u])
	{
		dfs2(v);
	}
}

void output()
{
	for (int i = N - 1; i >= 0; i--)
	{
		dfs2(order[i]);
		// cout << "from " << order[i] << endl;
		if (!st.empty())
		{
			int comsz = 0;
			while (!st.empty())
			{
				int v = st.top();
				st.pop();
				to[v] = comid;
				com[comid].push_back(v);
				comsz++;
			}
			sort(com[comid].begin(), com[comid].end());
			comid++;
		}
	}
}

int main()
{
	cin >> N >> M;
	for (int i = 0; i < M; i++)
	{
		int u, v;
		cin >> u >> v;
		edg[u].push_back(v);
		rev[v].push_back(u);
	}

	memset(vis, 0, sizeof(vis));
	
	for (int i = 1; i <= N; i++)
	{
		dfs1(i);
	}

	memset(vis, 0, sizeof(vis));

	output();

	cout << comid << endl;

	for (int i = 1; i <= N; i++)
	{
		if (viscom[to[i]]) { continue; }
		viscom[to[i]] = true;
		for (int v : com[to[i]]) { cout << v << " "; }
		cout << endl;
	}

	return 0;
}
```



### LCA 最小公共祖先 倍增算法

```c++
// P3379

typedef long long ll;
using namespace std;

const int maxn = 5e5 + 2000;
const int maxd = 31;
vector<int> edge[maxn];
int depth[maxn];
int n, m, s;
int father[maxn][maxd] = { 0 };

void dfs(int root, int fa)
{
	father[root][0] = fa;
	depth[root] = depth[fa] + 1;
	for (int i = 1; i < maxd; i++)
	{
		father[root][i] = father[father[root][i - 1]][i - 1];
	}
	for (int j : edge[root])
	{
		if (j != fa) { dfs(j, root); }
	}
}

int lca(int a, int b)
{
	if (depth[a] < depth[b]) { swap(a, b); }
	int delta = depth[a] - depth[b];
	for (int exp = 0; exp < 30; exp++)
	{
		if ((1 << exp) & delta)
		{
			a = father[a][exp];
		}
	}
	// cout << "depth, a = " << depth[a] << ", b = " << depth[b] << endl;
	if (a == b) { return a; }
	for (int exp = 29; exp >= 0; exp--)
	{
		if (father[a][exp] != father[b][exp])
		{
			a = father[a][exp];
			b = father[b][exp];
		}
	}
	return father[a][0];
}

int main()
{
	cin >> n >> m >> s;
	for (int i = 1; i < n; i++)
	{
		int u, v;
		scanf("%d%d", &u, &v);
		edge[u].push_back(v);
		edge[v].push_back(u);
	}
	depth[s] = 0;
	dfs(s, 0);
	for (int i = 0; i < m; i++)
	{
		int u, v;
		scanf("%d%d", &u, &v);
		printf("%d\n", lca(u, v));
	}
	return 0;
}
```

### 拓扑排序 + dp

```c++
 //P1113

typedef long long ll;
using namespace std;
int n;
const int maxn = 1e4 + 2000;
vector<int> out[maxn]; // out, store edge
int len[maxn];
int in_count[maxn] = { 0 }; // in, store count
int dp[maxn] = { 0 };
queue<int> que;

int toposort()
{
	int ans = 0;
	while (!que.empty())
	{
		int now = que.front();
		que.pop();
		int timeend = dp[now] + len[now];
		ans = max(ans, timeend);
		for (int v : out[now])
		{
			dp[v] = max(dp[v], timeend);
			in_count[v]--;
			if (in_count[v] == 0) { que.push(v); }
		}
	}
	return ans;
}
```

### Floyd

犯过的错：k没放在最外面，想想为什么要。

```c++
void floyd(int *edge[], int n)
{
	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			edge[i][j] = INT_MAX / 2;
		}
		edge[i][i] = 0;
	}

	for (int k = 1; k <= n; k++)
	{
		for (int i = 1; i <= n; i++)
		{
			for (int j = 1; j <= n; j++)
			{
				edge[i][j] = min(edge[i][j], edge[i][k] + edge[k][j]);
			}
		}
	}
}
```

### Bellman-Ford

注意外层 $\left| V \right|$ 次，内层 $\left| E \right|$ 次。

```cpp
struct Edge {
  int u, v, w;
};

vector<Edge> edge;

int dis[MAXN], u, v, w;
const int INF = 0x3f3f3f3f;

bool bellmanford(int n, int s) {
  memset(dis, 0x3f, sizeof(dis));
  dis[s] = 0;
  bool flag = false;  // 判断一轮循环过程中是否发生松弛操作
  for (int i = 1; i <= n; i++) {
    flag = false;
    for (int j = 0; j < edge.size(); j++) {
      u = edge[j].u, v = edge[j].v, w = edge[j].w;
      if (dis[u] == INF) continue;
      // 无穷大与常数加减仍然为无穷大
      // 因此最短路长度为 INF 的点引出的边不可能发生松弛操作
      if (dis[v] > dis[u] + w) {
        dis[v] = dis[u] + w;
        flag = true;
      }
    }
    // 没有可以松弛的边时就停止算法
    if (!flag) {
      break;
    }
  }
  // 第 n 轮循环仍然可以松弛时说明 s 点可以抵达一个负环
  return flag;
}
```

### Dijkstra

```cpp
void Dijkstra() {
    memset(vis, 0, sizeof(vis));
    memset(d, 0x3f, sizeof(d));
    priority_queue<pair<int, int> > q;
    q.push(make_pair(0, 1));
    d[1] = 0;
    while (q.size()) {
        int u = q.top().second;
        q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (vector<Edge>::iterator it = G1[u].begin(); it != G1[u].end(); it++) {
            int v = it->v, w = it->w;
            if (d[u] + w < d[v]) {
                d[v] = d[u] + w;
                q.push(make_pair(-d[v], v));
            }
        }
    }
}
```

### Dinic

```cpp
typedef long long ll;
using namespace std;

const int N = 10010, M = 200010;
const ll INF = 1e15;

class edge
{
public:
	int ed;
	ll len;
	int id;
	edge(int a, ll b, int i) { ed = a; len = b; id = i; }
};

vector <edge> e[N];

int n, m, S, T;
int dep[N], cur[N];

bool bfs()
{
	memset(dep, -1, sizeof dep);
	queue <int> q;
	q.push(S);
	dep[S] = 0;
	while (!q.empty())
	{
		int src = q.front();
		q.pop();
		for (int i = 0; i < e[src].size(); i = i + 1)
		{
			int dst = e[src][i].ed;
			if (dep[dst] == -1 && e[src][i].len)
			{
				dep[dst] = dep[src] + 1;
				q.push(dst);
			}
		}
	}
	memset(cur, 0, sizeof(cur));
	return (dep[T] != -1);
}

ll dfs(int st, ll limit)
{
	if (st == T)
		return limit;
	for (int i = cur[st]; i < e[st].size(); i = i + 1)
	{
		cur[st] = i;  // 当前弧优化
		int ed = e[st][i].ed;
		if (dep[ed] == dep[st] + 1 && e[st][i].len)
		{
			int t = dfs(ed, min(e[st][i].len, limit));
			if (t)
			{
				e[st][i].len -= t;
				e[ed][e[st][i].id].len += t;
				return t;
			}
			else
				dep[ed] = -1;
		}
	}
	return 0;
}

int dinic()
{
	int r = 0, flow;
	while (bfs())
	{
		while (flow = dfs(S, INF)) r += flow;
	}
	return r;
}

int main()
{
	cin >> n >> m >> S >> T;
	while (m--)
	{
		int u, v; ll len;
		cin >> u >> v >> len;
		int uid = e[u].size(); // 反向边的 id，方便访问
		int vid = e[v].size();
		e[u].push_back(edge(v, len, vid));
		e[v].push_back(edge(u, 0, uid));
	}
	cout << dinic();

	return 0;
}
```

### 二分图匹配的匈牙利算法

P3386

```c++
typedef long long ll;
using namespace std;

const int INF = INT_MAX;
const int maxn = 2e3;
vector<int> edge[maxn];
int vis[maxn] = { 0 }; int match[maxn] = { 0 };

bool dfs(int src, int mark)
{
	if (vis[src] == mark) { return false; }
	vis[src] = mark;
	for (int dst : edge[src])
	{
		if (match[dst] == 0 || dfs(match[dst], mark))
		{
			match[dst] = src;
			return true;
		}	
	}
	return false;
}

int N, M, E;

int main()
{
	cin >> N >> M >> E;
	for (int i = 1; i <= E; i++)
	{
		int u, v;
		cin >> u >> v;
		edge[u].push_back(v); 
	}
	int ans = 0;
	for (int i = 1; i <= N; i++)
	{
		if (dfs(i, i)) { ans++; }
	}
	cout << ans;
	return 0;
}
```



### 欧拉路径

P7771

```c++
#include <cstdio>
#include <iostream>
#include <climits>
#include <vector>
#include <algorithm>
#include <stack>

typedef long long ll;
using namespace std;

const int maxn = 2e5;
vector<int> edge[maxn];
int in[maxn] = { 0 }, out[maxn] = { 0 }, curr[maxn];
int N, M;

stack<int> st;

void dfs(int src)
{
	int len = (int)edge[src].size();
	for (int i = curr[src]; i < len; i = curr[src])
	{
		curr[src]++;
		dfs(edge[src][i]);
	}
	st.push(src);
}

int main()
{
	cin >> N >> M;
	for (int i = 1; i <= M; i++)
	{
		int u, v;
		cin >> u >> v;
		edge[u].push_back(v);
		out[u]++;
		in[v]++;
	}
	int start = 0;
	int endpt = 0;
	for (int i = 1; i <= N; i++)
	{
		if (in[i] + 1 == out[i])
		{
			if (!start) { start = i; }
			else { cout << "No"; return 0; }
		}
		else if (out[i] + 1 == in[i])
		{
			if (!endpt) { endpt = i; }
			else { cout << "No"; return 0; }
		}
		else if (in[i] != out[i]) { cout << "No"; return 0; }
		
		sort(edge[i].begin(), edge[i].end());
	}
	if ((start > 0)^(endpt > 0)) { cout << "No"; return 0; }
	if (start == 0) { start = 1; }
	
	dfs(start);
	
	while(!st.empty()) { cout << st.top() << " "; st.pop();
	}
	
	return 0;
} 
```



## 动态规划

### 背包问题

初始化：求最值时初始化为 0，如果要求必须消耗完则 dp[0] = 0，剩下赋为无穷值。

#### 0-1 背包

多重背包可以二进制优化为01背包问题。

```cpp
for (int i=1; i<=n; i++)
	for (int j=m; j>=w[i]; j--)
		f[j] = max(f[j], f[j-w[i]]+v[i]);
```

#### 完全背包（没有数量限制）

```cpp
for (int i=1; i<=n; i++)
	for (int j=w[i]; j<=m; j++)
		f[j] = max(f[j], f[j-w[i]]+v[i]);
```

### 状压dp

```c++
// “吃奶酪”

typedef long long ll;
using namespace std;

ll n;
const int maxn = 1e6 + 2000;
const int maxv = 15 + 2;
double dp[maxn][maxv] = { 0 };
ll last[maxn] = { 0 };
double cord[maxn][2] = { 0 };

double distance(int i, int j)
{
	return sqrt((cord[i][0] - cord[j][0]) * (cord[i][0] - cord[j][0]) + (cord[i][1] - cord[j][1]) * (cord[i][1] - cord[j][1]));
}

int compress_main()
{
	cin >> n;
	for (int i = 1; i <= n; i++)
	{
		cin >> cord[i][0] >> cord[i][1];
	}

	int inf = 1 << n;

	for (int i = 0; i < inf; i++)
	{
		for (int j = 0; j <= n; j++)
		{
			dp[i][j] = (double)(INT_MAX);
		}
	}

	for (int i = 1; i <= n; i++)
	{
		dp[1 << i - 1][i] = distance(0, i);
		// cout << "dp[" << (1 << i - 1) << "][" << i << "] = " << dp[1 << i - 1][i] << endl;
	}

	for (int i = 1; i < inf; i++)
	{
		// cout << "i = " << i << endl;
		for (int j = 1; j <= n; j++)
		{
			if ((1 << j - 1 & i) == 0) { continue; }
			for (int k = 1; k <= n; k++)
			{
				if ((j == k) || ((1 << k - 1 & i) == 0)) { continue; }
				// cout << "j = " << j << ", k = " << k << " from " << i - (1 << j - 1) << " to " << i << endl;
				dp[i][j] = min(dp[i][j], dp[i - (1 << j - 1)][k] + distance(j, k));
				// cout << "dp[" << i << "][" << j << "] = " << dp[i][j] << endl;
			}
		}
	}
	double ans = (double)(INT_MAX);
	for (int i = 1; i <= n; i++)
	{
		ans = min(ans, dp[inf - 1][i]);
	}
	printf("%.2lf", ans);
	return 0;
}

```

### 导弹拦截：最长不增子序列与 Dilworth 定理 (P1020)

```C++
typedef long long ll;
using namespace std;
const int maxn = 1e5 + 10;

int n, r1, r2;
int arr[maxn], l[maxn], h[maxn];
int main() {
    int i = 0;
    while (cin >> arr[i]) { i++; }
    n = i;
    l[0] = h[0] = arr[0]; r1 = r2 = 0;
    for (int i = 1; i < n; ++i) {
        if (l[r1] >= arr[i]) { // 求单调不增
            r1++; l[r1] = arr[i];
        }
        else *upper_bound(l, l + r1, arr[i], greater<int>()) = arr[i]; // 第一个小于 arr[i] 的数
        if (h[r2] < arr[i]) { // 求单调增 (Dilworth)
            r2++; h[r2] = arr[i];
        }
        else *lower_bound(h, h + r2, arr[i]) = arr[i]; // 第一个大于等于 arr[i] 的数
    } printf("%d\n%d", r1 + 1, r2 + 1); // 下标 -> 数量
    return 0;
}
```



## 表达式求值

用栈。

### 中缀表达式求值

需要两个栈来存储数字型元素和符号型元素，遇到符号时判断符号栈是否为空，为空直接入栈，不为空的前提下在判断运算符的优先级，再做处理。最后运算的结果保存在数字栈中，只需弹栈即可。

参考：[详细理解中缀表达式并实现-CSDN博客](https://blog.csdn.net/cout_s/article/details/118767818)

### 中缀表达式转后缀表达式

**从左到右**遍历中缀表达式的每个操作数和操作符。

- 当读到操作数时，立即把它输出，即成为后缀表达式的一部分；

- 若读到操作符，判断该符号与栈顶符号的优先级，

  - 若该符号优先级大于等于栈顶元素，则将该操作符入栈，

  - 否则就一次把栈中运算符弹出并加到后缀表达式尾端，直到遇到优先级低于该操作符的栈元素，然后把该操作符压入栈中。

- 如果遇到”(”，直接压入栈中，如果遇到一个”)”，那么就将栈元素弹出并加到后缀表达式尾端，但左右括号并不输出。最后，如果读到中缀表达式的尾端，将栈元素依次完全弹出并加到后缀表达式尾端。

### 中缀表达式转前缀表达式

**从右至左**扫描中缀表达式。

- 如果是操作数，则直接输出，作为前缀表达式的一个直接转换表达式Temp（最后，前缀表达式由该表达式翻转得到）；
- 如果是运算符，则比较优先级：
  - 若该运算符优先级大于等于栈顶元素，则将该运算符入栈；
  - 否则栈内元素出栈并加到Temp表达式尾端，直到遇到优先级低于该操作符的栈元素，然后把该操作符压入栈中。
- 遇到右括号直接压入栈中，如果遇到一个左括号，那么就将栈元素弹出并加到Temp表达式尾端，但左右括号并不输出。最后，若运算符栈中还有元素，则将元素一次弹出并加到Temp表达式尾端，最后一步是将Temp表达式翻转。



## 优化

### 树状数组

P3374

```c++
int const maxn = 6e5;
int N, M;
ll c[maxn];

ll lowbit(int x)
{
	return x & -x;
}

void add(int u, int v) // 单元素修改
{
	for (int i = u; i <= N; i += lowbit(i))
		c[i] += v;
}

ll sum(int u) // 直到该下标的元素和
{
	int ans = 0;
	for (int i = u; i > 0; i -= lowbit(i))
		ans += c[i];
	return ans;
}

int main()
{
	cin >> N >> M;
	for (int i = 1; i <= N; i++)
	{
		ll w;
		cin >> w;
		add(i, w);
	}
	for (int i = 1; i <= M; i++)
	{
		int op, x, y;
		scanf("%d%d%d", &op, &x, &y);
		if (op == 1)
		{
			add(x, y);
		}
		if (op == 2)
		{
			printf("%lld\n", sum(y) - sum(x - 1));
		}
	}
	return 0;
}
```

