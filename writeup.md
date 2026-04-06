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

### 4.2 The SGD Optimizer

A small learning rate (1e1) causes slow but steady decay, a moderate learning rate (1e2) converges much faster and reaches near-zero, while a large learning rate (1e3) causes the loss to diverge exponentially.

### 4.3 AdamW accounting

(a) Let $B$ = batch_size, $T$ = context_length, $L$ = num_layers, $H$ = num_heads, $d_m$ = d_model, $V$ = vocab_size, $d_{ff} = 4d_m$.

**Parameters**

Per transformer block:
- 2 RMSNorm weights: $2d_m$
- MHA (Q, K, V, O projections): $4d_m^2$
- FFN (W1, W2, W3): $3d_m d_{ff} = 12d_m^2$

Total per block: $2d_m + 16d_m^2$

Outside blocks (token embedding, final RMSNorm, lm_head): $2Vd_m + d_m$

$$P = L(2d_m + 16d_m^2) + 2Vd_m + d_m$$

$$\text{Memory}_{\text{params}} = 4P \text{ bytes}$$

**Activations** (per transformer block)

- RMSNorm (×2): $2 \cdot BTd_m$
- MHA: Q, K, V, weighted-sum output, output projection: $5 \cdot BTd_m$; attention scores + softmax output: $2 \cdot BHT^2$
- FFN: W1 output + SiLU + W2 output: $2 \cdot BT d_{ff} + BTd_m = 9 \cdot BTd_m$

Per block total: $16BTd_m + 2BHT^2$

Outside blocks: final RMSNorm ($BTd_m$), output embedding ($BTV$), cross-entropy ($BTV$)

$$\text{Memory}_{\text{activations}} = 4\bigl(L(16BTd_m + 2BHT^2) + BTd_m + 2BTV\bigr) \text{ bytes}$$

**Gradients**

Same shape as parameters:

$$\text{Memory}_{\text{gradients}} = 4P \text{ bytes}$$

**Optimizer state** (AdamW stores first and second moment vectors $m$, $v$ per parameter):

$$\text{Memory}_{\text{optimizer}} = 8P \text{ bytes}$$

**Total**

$$\text{Total} = 16P + 4\bigl(L(16BTd_m + 2BHT^2) + BTd_m + 2BTV\bigr) \text{ bytes}$$

(b) For GPT-2 XL: $L=48$, $d_m=1600$, $H=25$, $T=1024$, $V=50257$, $d_{ff}=6400$.

$$P = 48(2 \cdot 1600 + 16 \cdot 1600^2) + 2 \cdot 50257 \cdot 1600 + 1600 = 2{,}127{,}057{,}600$$

$$16P = 34{,}032{,}921{,}600 \text{ bytes} \approx 31.69 \text{ GB (model + gradients + optimizer)}$$

Activation memory per batch element:

$$4\bigl(48(16 \cdot 1024 \cdot 1600 + 2 \cdot 25 \cdot 1024^2) + 1024 \cdot 1600 + 2 \cdot 1024 \cdot 50257\bigr) = 15{,}517{,}753{,}344 \approx 14.45 \text{ GB}$$

Total memory as a function of batch size:

$$\text{Total} = 14.45 \cdot B + 31.69 \text{ GB}$$

For 80 GB: $B \leq \frac{80 - 31.69}{14.45} \approx 3.34$, so the maximum batch size is $B = 3$.

(c) AdamW performs a fixed number of elementwise operations per parameter: updating $m$ (3 FLOPs), updating $v$ (4 FLOPs), computing $\sqrt{v}+\epsilon$ (2 FLOPs), the parameter update (3 FLOPs), and weight decay (2 FLOPs) — roughly 14 FLOPs per parameter. Thus one AdamW step takes $O(P) \approx 14P$ FLOPs total. This is negligible compared to the forward/backward pass, which scales as $O(Pd_m)$ due to matrix multiplications.

(d) Following Kaplan et al. and Hoffmann et al., the total training FLOPs $\approx 6P \cdot D$, where $D$ is the total number of tokens processed and the factor of 6 accounts for the forward pass ($\approx 2P$ FLOPs/token) plus a backward pass twice as costly ($\approx 4P$ FLOPs/token).

$$D = B \times T \times \text{steps} = 1024 \times 1024 \times 400{,}000 \approx 4.19 \times 10^{11} \text{ tokens}$$

$$\text{Total FLOPs} = 6 \times 2.127 \times 10^9 \times 4.19 \times 10^{11} \approx 5.35 \times 10^{21}$$

At 50% MFU on an A100 (19.5 TFLOPs/s):

$$\text{Effective throughput} = 0.5 \times 19.5 \times 10^{12} = 9.75 \times 10^{12} \text{ FLOPs/s}$$

$$\text{Time} = \frac{5.35 \times 10^{21}}{9.75 \times 10^{12}} \approx 5.49 \times 10^{8} \text{ s} \approx 6{,}354 \text{ days} \approx 17.4 \text{ years}$$

