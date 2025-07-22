#include <bits/stdc++.h>
#include <algorithm>

using namespace std;


extern "C" {
    const static int m1 = 4, m2 = 4;
    const double V = 10000;
    int ins[16384], que[16384];
    class SS {
    private:
        int adjust_thres;
        int lim, num, real_n, t = 0;
        int n1, n2;
        float Threshold;
        double tot;
        double p;
        // Instance-specific state variables (moved from global)
        double alpha;
        double decay_importance;
        int batch_num;
        bool global_flip_bit;
        struct Bucket{
            int val[m1];
            float cnt[m1];
            int dic[m1];
            bool flip_bit;
        }*b;
        struct Bucket2{
            int val[m2];
            float cnt[m2];
            int dic[m2];
            int t[m2];
            bool flip_bit;
        }*b2;
    public:
        queue<uint32_t> hot_id;
        int Hash1(uint32_t x) {
            return x * 1000000007ll % n1;
        }
        int Hash2(uint32_t x) {
            return x * 1000000007ll % n2;
        }
        SS(float Threshold = 200, int lim = 130670, int adjust_thres=1): 
            Threshold(Threshold), lim(lim), adjust_thres(adjust_thres){
            tot = 0;
            num = 0;
            real_n = 0;
            // Initialize instance-specific state
            alpha = 1.000001;
            decay_importance = 1.0;
            batch_num = 0;
            global_flip_bit = false;
            printf("size: %d\n", lim);
            
            n1 = lim * 0.9;
            n2 = lim * 0.1;
            b = new Bucket[n1];
            b2 = new Bucket2[n2];
            for (int i = 1; i < lim; ++i)
                hot_id.push(i);
            for (int i = 0; i < n1; ++i) {
                memset(b[i].cnt, 0, sizeof(b[i].cnt));
                memset(b[i].dic, 0, sizeof(b[i].dic));
                b[i].flip_bit = 0;
            }
            for (int i = 0; i < n2; ++i) {
                memset(b2[i].cnt, 0, sizeof(b2[i].cnt));
                memset(b2[i].dic, 0, sizeof(b2[i].dic));
                memset(b2[i].t, 0, sizeof(b2[i].t));
                b2[i].flip_bit = 0;
            }
        }
        
        void set_alpha(double new_alpha) {
            alpha = new_alpha;
        }
        
        ~SS() {
            if (b != nullptr) {
                delete[] b;
                b = nullptr;
            }
            if (b2 != nullptr) {
                delete[] b2;
                b2 = nullptr;
            }
        }
        void reset() {
            printf("reset: %d\n", num);
            fflush(stdout);
            vector<pair<float, int*> > vec;
            for (int key = 0; key < n1; ++key) {
                if (b[key].flip_bit != global_flip_bit)
                    update_filp_bit1(key);
                for (int i = 0; i < m1; ++i) {
                    if (b[key].cnt[i] >= Threshold || b[key].dic[i])
                        vec.push_back(make_pair(b[key].cnt[i], &b[key].dic[i]));
                }
            }
            for (int key = 0; key < n2; ++key) {
                if (b2[key].flip_bit != global_flip_bit)
                    update_filp_bit2(key);
                for (int i = 0; i < m2; ++i) {
                    if (b2[key].cnt[i] >= Threshold || b2[key].dic[i])
                        vec.push_back(make_pair(b2[key].cnt[i], &b2[key].dic[i]));
                }
            }
            sort(vec.begin(), vec.end());
            int l = vec.size();
            printf("reset***: %d, %d, %ld\n", l, lim, hot_id.size());
            fflush(stdout);
            
            // 添加边界检查，防止数组越界
            if (l <= lim) {
                // 如果 vec 大小不够，直接返回，不修改 Threshold
                printf("Warning: vec size %d <= lim %d, skipping reset\n", l, lim);
                return;
            }
            
            for (int i = 0; i <= l - lim; ++i) {
                if ((*vec[i].second) != 0) {
                    hot_id.push(*vec[i].second);
                    *vec[i].second = 0;
                }
            }
            for (int i = l - lim + 1; i < l; ++i) {
                if ((*vec[i].second) == 0){
                    *vec[i].second = hot_id.front();
                    hot_id.pop();
                }
            }
            Threshold = vec[l - lim].first;
            cout << "threshold: " << Threshold << endl;
            fflush(stdout);
            real_n = lim;
        }

        int query(uint32_t val) {
            int key = Hash1(val);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == val) {
                    if (b[key].dic[i]) return -b[key].dic[i];
                }
            }
            int v = queryLRU(val);
            if (v != 0) return -v;
            return val;
        }

        void update_filp_bit1(int key) {
            b[key].flip_bit = global_flip_bit;
            for (int i = 0; i < m1; ++i)
                b[key].cnt[i] /= V;
        }

        void update_filp_bit2(int key) {
            b2[key].flip_bit = global_flip_bit;
            for (int i = 0; i < m2; ++i)
                b2[key].cnt[i] /= V;
        }

        bool Query(uint32_t x) {
            int key = Hash1(x);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == x) {
                    return 1;
                }
            }
            return 0;
        }

        int queryLRU(uint32_t x) {
            int key = Hash2(x);
            for (int i = 0; i < m2; ++i) {
                if (b2[key].val[i] == x) {
                    return b2[key].cnt[i];
                }
            }
            return 0;
        }
        bool insertLRU(uint32_t x, float count = 1) {
            int key = Hash2(x), p = -1, t_min = 1e9, id = 0;
            if (b2[key].flip_bit != global_flip_bit) update_filp_bit2(key);
            for (int i = 0; i < m2; ++i) {
                if (b2[key].val[i] == x) {
                    b2[key].t[i] = ++t;
                    b2[key].cnt[i] += count;
                    // cout << "cnt: " << b2[key].cnt[i] << endl;
                    if (b2[key].cnt[i]>= Threshold && b2[key].cnt[i] - count < Threshold) {
                        real_n ++;
                    }
                    if (b2[key].cnt[i] >= Threshold && !hot_id.empty() && !b2[key].dic[i]) {
                        b2[key].dic[i] = hot_id.front();
                        id = 1;
                        ++num;
                        hot_id.pop();
                    }
                    if (b2[key].cnt[i] >= Threshold) {
                        Insert(b2[key].val[i], b2[key].cnt[i], b2[key].dic[i]);
                        b2[key].val[p] = 0;
                        b2[key].t[p] = 0;
                        b2[key].cnt[p] = 0;
                        b2[key].dic[p] = 0;
                    }
                    return id;
                }
                if (t_min > b2[key].t[i]) 
                    t_min = b2[key].t[i], p = i;
            }
            if (b2[key].cnt[p] >= 5) {
                Insert(b2[key].val[p], b2[key].cnt[p], b2[key].dic[p]);
            }
            b2[key].val[p] = x;
            b2[key].t[p] = ++t;
            b2[key].cnt[p] = 1;
            b2[key].dic[p] = 0;
            return 0;
        }
        int insert(uint32_t x, float count = 1) {
            int key = Hash1(x), id = 0;
            if (b[key].flip_bit != global_flip_bit) update_filp_bit1(key);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == x) {
                    b[key].cnt[i] += count;
                    float cnt = b[key].cnt[i];
                    if (cnt >= Threshold && cnt - count < Threshold) {
                        real_n ++;
                    }
                    if (cnt >= Threshold && !b[key].dic[i] && !hot_id.empty()) {
                        b[key].dic[i] = hot_id.front();
                        id = 1;
                        ++num;
                        hot_id.pop();
                    }
                    return id;
                }
            }
            return insertLRU(x, count);
        }
        void Insert(uint32_t x, float count = 1, int Dic = 0) {
            int key = Hash1(x);
            int min_index = -1;
            float min_cnt = 1e9;
            if (b[key].flip_bit != global_flip_bit) update_filp_bit1(key);
            for (int i = 0; i < m1; ++i) {
                if (b[key].val[i] == 0) {
                    b[key].val[i] = x;
                    b[key].cnt[i] = count;
                    b[key].dic[i] = Dic;
                    return;
                }
                if (b[key].cnt[i] < min_cnt){
                    min_cnt = b[key].cnt[i];
                    min_index = i;
                }
            }
            if (!b[key].dic[min_index]){
                b[key].cnt[min_index] += count;
                b[key].val[min_index] = x;
                b[key].dic[min_index] = Dic;
            } else {
                hot_id.push(Dic);
            }
        }
        int* batch_query(uint32_t *data, int len) {
            for (int i = 0; i < len; ++i) {
                que[i] = query(data[i]);
            }
            return que;
        }
        int* batch_insert(uint32_t *data, int len) {
            ++batch_num;
            decay_importance *= alpha;
            if (decay_importance > V) {
                decay_importance /= V;
                Threshold = max((float)(Threshold / V), 1.0f);  // 防止 Threshold 过小
                global_flip_bit ^= 1;
            }
            // cout << "real: " << real_n << " lim: " << lim << " threshold: " << Threshold << endl;
            if (real_n > lim * 1.2 && adjust_thres) reset();
            for (int i = 0; i < len; ++i) {
                ins[i] = insert(data[i], 1);
            }
            return ins;
        }
        int* batch_insert_val(uint32_t *data, float *v, int len) {
            ++batch_num;
            // cout << "real: " << real_n << " " << lim << " " << decay_importance << " " << alpha << endl;
            decay_importance *= alpha;
            if (decay_importance > V) {
                decay_importance /= V;
                Threshold = max((float)(Threshold / V), 1.0f);  // 防止 Threshold 过小
                global_flip_bit ^= 1;
            }
            // cout << "real: " << real_n << " lim: " << lim << endl;
            if (real_n > lim * 1.2 && adjust_thres) reset();
            for (int i = 0; i < len; ++i) {
                ins[i] = insert(data[i], v[i]);
            }
            return ins;
        }
    }*ss;
    float cntm[16384];
    
    int* batch_query(uint32_t *data, int len) {
        return ss->batch_query(data, len);
    }
    int* batch_insert(uint32_t *data, int len) {
        return ss->batch_insert(data, len);
    }
    int* batch_insert_val(uint32_t *data, float *v, int len) {
        return ss->batch_insert_val(data, v, len);
    }
    void init(int n, int Threshold, int adjust_thres, double alp){
        ss = new SS((float)Threshold, n, adjust_thres);
        cout << "alp: " << alp << endl;
        ss->set_alpha(alp);
    }
    
    void cleanup() {
        if (ss != nullptr) {
            delete ss;
            ss = nullptr;
        }
    }
    
    double analyse(uint32_t* data, int len) {
        int ans = 0;
        for (int i = 0; i < len; ++i) {
            if (ss->query(data[i]) < 0) ++ans;
        }
        return 1.0 * ans / len;
    }
    double analyse1(uint32_t* data, int len) {
        int ans = 0;
        for (int i = 0; i < len; ++i) {
            if (ss->Query(data[i])) ++ans;
        }
        return 1.0 * ans / len;
    }
}


int main() {
    return 0;
}