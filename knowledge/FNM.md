To restate the entire FNM (False Negative Mitigation) method from Paper 2 ("Towards Mitigation of False Negatives in Text-to-Image Person Re-identification"), below is a detailed analysis of the complete architecture, algorithmic steps, and accompanying mathematical formulas based on the original document.

The FNM method consists of 3 main components: **Multi-modal Feature Representation**, **Mitigating the False Negatives**, and **Learning with Momentum Contract**.
Aim to handle False Negative: This is the phenomenon where image samples possess highly similar semantic content to the text query (sharing a common meaning) but are classified and treated by the system as "negative" (non-matching) samples. False negatives primarily originate from two practical limitations of datasets:

* Person ID labeling errors: The same person appearing across different cameras is mistakenly assigned multiple different identities (IDs), causing the system to assume they are unrelated individuals.

* Ambiguous/generic text descriptions: A text segment describing physical appearance could correctly apply to many different people in the dataset (e.g., the description "a man wearing a white shirt and carrying a black backpack" might match multiple images with different IDs).

---

### 1. Multi-modal Feature Representation

The model utilizes pre-trained CLIP encoders to extract image and text features at both levels: Global and Local.

**1.1. Global-level Features**
* **Image:** The input image $I_i \in \mathbb{R}^{h \times w \times c}$ is divided into $nv = \frac{h \times w}{p^2}$ patches. Through the Vision Transformer (ViT) encoder, we obtain a sequence of tokens: $f_i^v = \{v_i^g, v_{i1}, v_{i2}, ..., v_{inv}\}$. Here, the first [CLS] token $v_i^g$ is used as the **global image feature**.
* **Text:** The text segment $T_i$ is BPE-encoded to form a sequence of tokens: $f_i^t = \{t_i^{sos}, t_{i1}, t_{i2}, ..., t_{int}, t_i^{eos}\}$. The final [EOS] token, $t_i^{eos}$, is used as the **global text feature** $t_i^g$.

**1.2. Dual-Level Local Features**
Instead of solely relying on global features, the algorithm selects the local tokens that carry the most useful information through the weights of the self-attention layer (attention map).

* **Token Selection:** Select the Top-K tokens with the highest attention weights, where $K = R \times nv$ ($R$ being the token selection ratio). The selected tokens form the set $\hat{f}_i^v = \{v_{ik_1^v}, v_{ik_2^v}, ..., v_{ik_n^v}\}$ for the image, and $\hat{f}_i^t$ for the text.
* **Feature Aggregation:** Pass these tokens through a residual-like block to obtain the final local features ($v_i^l$ for the image and $t_i^l$ for the text):
    $$t_i^l = \text{MaxPool}(\sigma(\text{BN}(W_1^t \hat{f}_i^t) + W_2^t \hat{f}_i^t))$$
    $$v_i^l = \text{MaxPool}(\sigma(\text{BN}(W_1^v \hat{f}_i^v) + W_2^v \hat{f}_i^v))$$
    (Where: MaxPool is the maximum pooling function, $\sigma$ is the ReLU activation function, BN is Batch Normalization, and $W$ denotes linear layers).

---

### 2. Mitigating the False Negatives

**2.1. False Negative Detection**
The objective is to calculate the probability that a negative sample is actually a false negative using Bayes' theorem.

* **Step 1: Collect similarity distribution:** The system collects the sets of similarity scores for positive pairs $S^+$ and negative pairs $S^-$:
    $$S^+ = [s_1^+, s_2^+, \dots, s_i^+, \dots]$$
    $$S^- = [s_1^-, s_2^-, \dots, s_i^-, \dots]$$
* **Step 2: Gaussian distribution modeling:** These two sets are modeled into two probability density functions following a normal distribution with mean ($\mu$) and standard deviation ($\sigma$):
    $$f_+(s) = \frac{1}{\sigma_+ \sqrt{2\pi}} e^{\left[-\frac{(s-\mu_+)^2}{2(\sigma_+)^2}\right]}$$
    $$f_-(s) = \frac{1}{\sigma_- \sqrt{2\pi}} e^{\left[-\frac{(s-\mu_-)^2}{2(\sigma_-)^2}\right]}$$
* **Step 3: Calculate false negative probability ($P(e|s)$):** Let $e$ be the event that a sample labeled as negative is actually a false negative. According to Bayes' theorem, the probability of event $e$ occurring given a similarity score $s$ is:
    $$P(e|s) = \frac{P(e) \int_{s}^{s+\epsilon} f_+(t) dt}{\int_{s}^{s+\epsilon} f(t) dt}$$
    As $\epsilon \to 0$ and applying the law of total probability, the equation is simplified into the posterior formula:
    $$P(e|s) = \frac{p \cdot f_+(s)}{p \cdot f_+(s) + (1-p) \cdot f_-(s)}$$
    (Where: $p$ is the prior probability, representing the frequency of false negatives occurring among negative samples).
* **Step 4: Thresholding decision:** Using a threshold $\theta$:
    * If $P(e|s) \ge \theta \rightarrow$ Identified as a **False negative**.
    * If $P(e|s) < \theta \rightarrow$ Identified as a **True negative**.

**2.2. False Negative Mitigation Loss ($L_{fnm}$)**
Instead of discarding false negative samples, the algorithm establishes an **adaptive margin ($\rho$)** that gradually decreases as the probability of a false negative increases, preventing that sample from being heavily penalized. The loss function for a mini-batch of size $B$ is:
$$L_{fnm} = - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^B \exp((\text{sim}(v_i, t_j) + \rho(v_i, t_j))/\tau)} - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^B \exp((\text{sim}(t_i, v_j) + \rho(t_i, v_j))/\tau)}$$

The margin adjustment mechanism $\rho(v_i, t_j)$ is defined as follows:
$$\rho(v_i, t_j) = \begin{cases} \gamma \cdot \frac{1 - r(v_i, t_j)}{1 - \theta}, & \text{if } r(v_i, t_j) \ge \theta, i \neq j \text{ (false negative)} \\ \gamma, & \text{if } r(v_i, t_j) < \theta, i \neq j \text{ (true negative)} \\ 0, & \text{if } i = j \text{ (positive)} \end{cases}$$
(Where: $\tau$ is the temperature parameter, $\gamma$ is the fixed margin, and $r(v_i, t_j)$ is precisely the probability $P(e|s)$ calculated above).

This loss function is aggregated for both the global ($L_{fnm}^g$) and local ($L_{fnm}^l$) levels:
$$L_{fnm} = L_{fnm}^l + L_{fnm}^g$$

---

### 3. Learning with Momentum Contractive Module (MoC)

Because estimating the Gaussian distribution (Step 2.1) requires a massive amount of sample points that a standard mini-batch cannot sufficiently provide, the algorithm designs the additional MoC module.

* **Four Storage Queues:** The system maintains 4 queues to store features from previous batches. This includes: 2 global feature queues ($Q_v^g$ for images, $Q_t^g$ for text) and 2 local feature queues ($Q_v^l$ for images, $Q_t^l$ for text). These queues operate on a First-In-First-Out (FIFO) mechanism.
* **Encoder Update via Exponential Moving Average (EMA):** To ensure the consistency of the features stored in the queues, the feature extractors used to push features into the queues are not updated through standard back-propagation, but are instead updated smoothly via a momentum parameter $m \in [0, 1)$:
    $$w_k^v = m \cdot w_k^v + (1-m) \cdot w^v$$
    $$w_k^t = m \cdot w_k^t + (1-m) \cdot w^t$$
    (Where: $w_k^v, w_k^t$ are the momentum network parameters; $w^v, w^t$ are the original network parameters).

Thanks to this MoC module, the amount of similarity scores collected in each training epoch becomes extremely abundant, helping the GMM (Gaussian Mixture Model) plot the two probability density curves for positive and negative samples with almost absolute accuracy.