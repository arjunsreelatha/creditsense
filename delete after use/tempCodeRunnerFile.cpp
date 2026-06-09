#include <bits/stdc++.h>
using namespace std;

int main() {

    int n;
    cin>> n;
    string s = "HelloWorld";
    string ans = s.substr(0, n);
    ans.append(s.susbstr(n, s.size() - n));
    cout << ans << endl;
}