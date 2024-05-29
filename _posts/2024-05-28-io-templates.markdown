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

当 $p$ 是质数时，$a^{p-1} \equiv 1\quad (\text{mod }p)$，$a \times a^{p-2} \equiv 1\quad (\text{mod }p)$，因此 $a$ 在 $\text{mod }p$ 意义下乘法逆元为 $a^{p-1}$。常结合快速幂求解。

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



## 图论

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

	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			for (int k = 1; k <= n; k++)
			{
				edge[i][j] = min(edge[i][j], edge[i][k] + edge[k][j]);
			}
		}
	}
}
```



## 动态规划

### 背包问题

初始化：求最值时初始化为 0，如果要求必须消耗完则 dp[0] = 0，剩下赋为无穷值。

#### 0-1 背包

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

