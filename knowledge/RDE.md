To present the complete details of the RDE (Robust Dual Embedding) method from the paper "Noisy-Correspondence Learning for Text-to-Image Person Re-identification," below is a comprehensive breakdown ranging from the problem statement and model architecture to the detailed mathematical formulas based on the original document.

The RDE method consists of two main components: the **Cross-modal Embedding Model** and **Robust Similarity Learning**.
Aim to handle False Positive (or False Positive Pairs - FPPs): This is the phenomenon where an image and text pair that actually do not match (which should be a negative pair) is incorrectly labeled as matching (a positive pair) for training. In the literature, this phenomenon is referred to as "Noisy Correspondence" (NC). The causes typically stem from degraded image quality (influenced by pedestrian posture, camera angles, or lighting conditions) or from manual human annotation errors.

---

### 1. Problem Statement

The goal of TIReID (Text-to-Image Person Re-identification) is to retrieve a person's image from a gallery set that matches a given text query.

Let the image set be $\mathcal{V} = \{I_i, y_i^p, y_i^v\}_{i=1}^{N_v}$ and the text set be $\mathcal{T} = \{T_i, y_i^v\}_{i=1}^{N_t}$. Here, $y_i^p$ is the person's identity (ID) label, and $y_i^v$ is the image label.

The training image-text pair dataset is $\mathcal{P} = \{(I_i, T_i), y_i^v, y_i^p\}_{i=1}^N$.

A binary correspondence label $l_{ij} \in \{0, 1\}$ is defined to indicate the matching degree of the pair $(I_i, T_j)$. If $l_{ij} = 1$, the pair matches (positive); if $l_{ij} = 0$, they do not match (negative).

**The Problem:** Due to noise, some pairs that actually do not match ($l_{ij} = 0$) are incorrectly labeled as matching ($l_{ij} = 1$), creating Noisy Correspondences (NCs).

---

### 2. Cross-modal Embedding Model

The authors utilize a pre-trained CLIP model as the feature extractor for both modalities.

**2.1. Token Representations**
* **For the image $I_i$:** The visual encoder $f_v$ divides the image into patches and encodes them into a token sequence of length $N_\circ + 1$:
    $$V_i = f_v(I_i) = \{v_i^g, v_{i1}, v_{i2}, \dots, v_{iN_\circ}\}^\top \in \mathbb{R}^{(N_\circ+1) \times d}$$
    Where $v_i^g$ is the global feature (from the [CLS] token), and $v_{ij}$ are the patch-level local features.
* **For the text $T_i$:** The text encoder $f_t$ processes the word sequence into a token sequence of length $N_\diamond + 2$:
    $$T_i = f_t(T_i) = \{t_{is}, t_{i1}, \dots, t_{iN_\diamond}, t_i^e\}^\top \in \mathbb{R}^{(N_\diamond+2) \times d}$$
    Where $t_i^e$ (from the [EOS] token) is used as the global text feature.

**2.2. Dual Embedding Modules**
To capture both global information and fine-grained details, the model employs two levels of embedding:
* **Basic Global Embedding (BGE):** Calculates the similarity score directly via cosine similarity between the two global tokens:
    $$S_{ij}^b = \frac{(v_i^g)^\top t_j^e}{\|v_i^g\| \|t_j^e\|}$$
* **Token Selection Embedding (TSE):** To capture detailed associations, the algorithm extracts correlation weights from the self-attention map in the final Transformer layer of CLIP. For images, the attention map is $A_i^v \in \mathbb{R}^{(1+N_\circ) \times (1+N_\circ)}$; the first row (the weights of the [CLS] token with other tokens) is taken as the selection score: $a_i^v = A_i^v[0, 1:N_\circ+1]$. For text, the attention map is $A_i^t$, and similarly, $a_i^t = A_i^t[0, 1:N_\diamond+1]$ is extracted. Afterward, the system retains the top-K tokens with the highest $a_i$ scores based on a selection ratio $R$ (e.g., $R=0.3$). These selected local tokens are passed through a residual block:
    $$v_i^{tse} = \text{MaxPool}(\text{MLP}(\hat{V}_i^s) + \text{FC}(\hat{V}_i^s))$$
    $$t_i^{tse} = \text{MaxPool}(\text{MLP}(\hat{T}_i^s) + \text{FC}(\hat{T}_i^s))$$
    (Where $\hat{V}_i^s$ and $\hat{T}_i^s$ are the L2-normalized tokens). The similarity of the local features is: $S_{ij}^t = \text{Cosine}(v_i^{tse}, t_j^{tse})$.

---

### 3. Robust Similarity Learning

**3.1. Confident Consensus Division (CCD)**
Based on the "memorization" effect of Neural Networks (clean data is learned first, leading to a lower loss), CCD uses the loss values to identify incorrectly labeled data.
* **Step 1:** Calculate the loss for each sample pair in a batch:
    $$\ell(\mathcal{M}, \mathcal{P}) = \{\ell_i\}_{i=1}^N = \{L(I_i, T_i)\}_{i=1}^N$$
* **Step 2:** Input this set of losses into a 2-component Gaussian Mixture Model (GMM) representing clean and noisy distributions. The EM algorithm estimates the distribution and calculates the posterior probability $p(k|\ell_i)$, where $k=0$ denotes the "clean" component (low loss).
* **Step 3:** Set a threshold $\delta = 0.5$ to divide the data into two sets:
    $$P_c = \{(I_i, T_i) \mid p(k=0|\ell_i) > \delta\} \quad \text{(Clean set)}$$
    $$P_n = \{(I_i, T_i) \mid p(k=0|\ell_i) \le \delta\} \quad \text{(Noisy set)}$$
* **Step 4:** Obtain the consensus from both the BGE and TSE network branches to establish absolute confidence:
    * **Confident clean set:** $\hat{P}_c = P_c^{bge} \cap P_c^{tse}$
    * **Confident noisy set:** $\hat{P}_n = P_n^{bge} \cap P_n^{tse}$
    * **Uncertain set:** $\hat{P}_u = P - (\hat{P}_c \cup \hat{P}_n)$
* **Step 5:** Label Recalibration: The original label $l_{ii}$ is corrected to a new label $\hat{l}_{ii}$:
    $$\hat{l}_{ii} = \begin{cases} 1, & \text{if } (I_i, T_i) \in \hat{P}_c \\ 0, & \text{if } (I_i, T_i) \in \hat{P}_n \\ \text{Rand}(\{0, 1\}), & \text{if } (I_i, T_i) \in \hat{P}_u \end{cases}$$

**3.2. Triplet Alignment Loss (TAL)**
The traditional TRL (Triplet Ranking Loss) formula is: $L_{trl} = [m - S^+ + S^-_{hardest}]_+$. This function can easily cause model collapse if the hardest negative sample ($S^-_{hardest}$) is actually noise. Therefore, RDE proposes the **TAL (Triplet Alignment Loss)**, which uses an upper bound to evaluate all negative samples instead of just taking the hardest one:
$$L_{tal}(I_i, T_i) = \left[ m - S_{i2t}^+(I_i) + \tau \log \left( \sum_{j=1}^K q_{ij} \exp(S(I_i, T_j) / \tau) \right) \right]_+ + \left[ m - S_{t2i}^+(T_i) + \tau \log \left( \sum_{j=1}^K q_{ji} \exp(S(I_j, T_i) / \tau) \right) \right]_+$$
Where:
* $m$ is the margin.
* $\tau$ is the temperature to control the hardness.
* $[x]_+ = \max(x, 0)$.
* $q_{ij} = 1 - l_{ij}$ (used to select only negative samples, zeroing out positive samples in the log-sum).
* $S_{i2t}^+$ is the weighted average of positive similarities (to handle cases where one person ID has multiple matching pairs in a single batch).

**Lemma 1:** Proves that TAL is the upper bound of TRL:
$$L_{trl}(I_i, T_i) \le L_{tal}(I_i, T_i)$$
Thanks to this relaxation, TAL distributes gradients smoothly across all negative samples ($j \neq i$) instead of concentrating all gradients onto a single hardest sample:
$$\frac{\partial L_{tal}}{\partial v_i} = \sum_{j \neq i} \beta_j (t_j - t_i)$$
where $\beta_j = \frac{\exp(v_i^\top t_j / \tau)}{\sum_{k \neq i} \exp(v_i^\top t_k / \tau)}$. This mechanism ensures the model neither collapses due to noise nor loses focus on learning the hard samples.

**3.3. Training and Inference**
During training, RDE applies the recalibrated labels ($\hat{l}_{ii}$) from CCD to the TAL function. The final total loss in a mini-batch of size $K$ is the sum of the losses from the BGE branch ($L_b$) and the TSE branch ($L_t$):
$$L_m = \sum_{i=1}^K \hat{l}_{ii} (L_b(I_i, T_i) + L_t(I_i, T_i))$$

During inference (testing), the final similarity score used for ranking an image-text pair is calculated as the average of the results from both embedding modules:
$$S = \frac{S^b + S^t}{2}$$



