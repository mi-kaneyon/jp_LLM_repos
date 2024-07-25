# InternLM2.5 7B Japanese script samples
- InternLM is originally Shanghai AI laboratory product.
- But Its Japanese level is acceptable for native Japanese.

# chat 

```
python lshen_chat.py

```

# person detect and evaluate feature

```
python cam_lshen.py

```
## Output sample

```

人物検出: この人物の特徴を説明してください。この人物は、「この人は、私の息子の友人の友人であり、私の息子の友人は年齢が少ない息子の友人であり、私の息子は年齢が少ない息子であり、私は年齢が少ない息子の友人から知っています。」という文脈で、人物の種類を分けられると、それは私の息子の友人の友人であり、私の息子の友人は年齢が少ない息子の友人であり、私の息子は年齢が少ない息子であり、私は年齢が少ない息子の友人から知っています。

この人物の特徴を説明するには、以下の点があります。

1. **関係の階層**：この人物は、あなたの息子の友人の中で二度の友人であり、この関係は年齢が少ない关系的層次の中でいくつかの層を涉る。

2. **年齢の関係**：この人物はあなたの息子の友人が年齢が少ない方に属し、年齢が少ない息子の友人という特定のグループに属する。

3. **関連する人々の年齢**：この人物があなたの息子の友人という人々と同じくらいの年齢にないという意味にして、あなたの息子の友人が年齢が少ない方に属する。

4. **連絡先**：この人物はあなたの息子の友人から知っていることを示している。これは、あなたの息子の友人を通じてこの人物について情報が共有されていることを意味し、互いに
人物検出: この人物の特徴を説明してください。

この人物は、1970年の米国選挙で、民主党のメイン人気サイドに所属し、または従事した人々には、その選挙で選挙戦を勝つにはどのようになりますかという意見を持つ人々がいました。この人物は選挙が勝つためには、メイン人気サイドに所属し、従事する必要があるという意見を持つ人々に、特定の選挙戦術を示しました。

この人物の選挙戦術は、民主党のメイン人気サイドに属于し、該サイドの活動を従事することで、選挙戦を勝つにはかつて必要なということを指します。この人物が示す選挙戦術は、過去に民主党はメイン人気サイドを指しますが、これは時々変わることがあります。

この人物が示す選挙戦術が、民主党の中でメイン人気サイドという概念が明確である時にのみ有効であることも指している可能性があります。民主党の選挙戦術は時々変わり、メイン人気サイドも変わります。

この人物が示す選挙戦術は、民主党の選挙戦術の一部であり、民主党は選挙戦術に富していることを指します。これは、過去の民主党の選挙戦術を確認し、民主党の選挙戦術の変化を見ることによって理解することができます。


```



## GPU VRAM usage 
```

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.29.06              Driver Version: 545.29.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        Off | 00000000:10:00.0  On |                  N/A |
| 61%   65C    P2             178W / 350W |  16530MiB / 24576MiB |     30%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2104      G   /usr/lib/xorg/Xorg                          358MiB |
|    0   N/A  N/A      2282      G   /usr/bin/gnome-shell                         90MiB |
|    0   N/A  N/A      4352      G   ...irefox/4539/usr/lib/firefox/firefox      138MiB |
|    0   N/A  N/A      5485      G   ...seed-version=20240723-180133.874000       68MiB |
|    0   N/A  N/A     77106      C   python                                    15846MiB |
+---------------------------------------------------------------------------------------+
```