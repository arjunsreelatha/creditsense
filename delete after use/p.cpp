#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W;
    long long K;
    cin >> H >> W >> K;

    vector<string> S(H);
    for (int i = 0; i < H; i++) {
        cin >> S[i];
    }

    vector<vector<int>> A(H, vector<int>(W));

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            A[i][j] = S[i][j] - '0';
        }
    }

    long long ans = 0;

    vector<int> col(W);

    for (int top = 0; top < H; top++) {

        fill(col.begin(), col.end(), 0);

        for (int bottom = top; bottom < H; bottom++) {

            for (int c = 0; c < W; c++) {
                col[c] += A[bottom][c];
            }

            unordered_map<long long, long long> freq;
            freq[0] = 1;

            long long pref = 0;

            for (int c = 0; c < W; c++) {
                pref += col[c];

                if (freq.count(pref - K))
                    ans += freq[pref - K];

                freq[pref]++;
            }
        }
    }

    cout << ans << '\n';
}