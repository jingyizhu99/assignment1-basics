### 2.1 Understanding Unicode

(a) It's the NULL character

(b) The string representation of chr(0) is '\x00', while the printed
representation is not visible

(c) The print statement renders text and shows "this is a teststring", while the
string representation prints invisible character "this is a test\x00string"

### 2.2 Unicode Encodings

(a) 1. utf-8 is dominant for web texts 2. utf-8 encoded bytes are more
space-efficient, where it uses 1 byte for ascii chars, comparing with utf-16 (2
bytes) and utf-32 (4 bytes). 3. UTF-8 naturally aligns with byte-level
processing, while UTF-16 and UTF-32 introduce complexity during grouping common
bytes.

(b) Example: café will be decoded as 'cafÃ©'. This function decodes the input
utf-8 strings one byte at a time, while the correct way is to decode the whole
string together.

(c) 0xC0 0xAF is a 2-byte sequence that does not decode to any Unicode character
in UTF-8 because it violates the encoding rules (overlong form)

### 2.5 BPE Training on TinyStories

(a) Training took around 8 minutes, used 5GB RAM. The longest token is
"accomplishment".

(b) Pretokenization takes most of the time & resource.


### 3.6 The Full Transformer LM

(a) Each transformer block has 2 * 1600 + 4 * (1600 * 1600) + 3 * (1600 * 6400) = 40,963,200 trainable parameters. 48 layers of them gives us 1,966,233,600 parameters, and those of embedding, norm and linear block, we have 2,127,057,600 parameters in total. They requires around 8.51 GB memory to load.

(b) MHA: $L(8BTd_m^2 + 4BT^2d_m)$, 
FFN: $L(6BTd_md_ff)$, 
lm_head: $2BTd_mV$
-> Total: $L(8BTd_m^2 + 4BT^2d_m + 6BTd_md_ff) + 2BTd_mV$
    = 48 * (20,971,520,000 + 6,710,886,400 + 62,914,560,000) + 164,682,137,600
    = 4,513,336,524,800

(c) All FFN layers combined has around 3T FLOPs. This layer require the most FLOPs

(d) Using $T=1024$, $B=1$, $V=50257$, $d_{ff}=4d_m$:

| Component | GPT-2 Small (L=12, d=768) | GPT-2 Medium (L=24, d=1024) | GPT-2 Large (L=36, d=1280) |
|-----------|--------------------------|------------------------------|---------------------------|
| MHA       | 27.6%                    | 29.9%                        | 30.0%                     |
| FFN       | 49.8%                    | 59.9%                        | 64.2%                     |
| lm_head   | 22.6%                    | 10.2%                        | 5.8%                      |
| **Total** | 350B FLOPs               | 1.03T FLOPs                  | 2.26T FLOPs               |

As model size increases, the FFN takes up proportionally more FLOPs because its cost scales as $d_m^2$ (since $d_{ff} = 4d_m$), while MHA's cost also scales as $d_m^2$ but includes a smaller $T^2$ term that becomes relatively less significant. The lm_head shrinks proportionally because its FLOPs scale as $d_m \cdot V$ (linear in $d_m$) compared to the $d_m^2$ scaling of MHA and FFN.

(e) Increasing context length from 1024 to 16384 (16x) increases total FLOPs by 33.1x (from 4.5T to 149.5T), since the MHA attention matmuls scale as $T^2$ while FFN and lm_head scale linearly with $T$. MHA becomes the dominant component, growing from 29.4% to 65.9% of total FLOPs, while FFN drops from 66.9% to 32.3%, as the quadratic $T^2$ attention cost overwhelms the linear FFN cost.
