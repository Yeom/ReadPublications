# Neural Machine Translation with Monolingual Translation Memory
- code : https://github.com/jcyk/copyisallyouneed

## Conclusion

Conclusion을 통해 이 논문의 저자들이 기여한 부분들을 확인해본다.

1. Effective approach that augments NMT models with monolingual TM
2. Task-specific cross-lingual memory retriever can be learned by end-to-end MT training
3. Achieves new S.O.T.A results
4. Large gains in low-resource scenarios where the bilingual data is limited
5. Specialize a NMT model for specific domains without further training

## Introduction

Non-parametric 메모리를 이용해 parametric 신경망 모델을 강화하는 방법은 최근에 많이 연구되고 있는데, 이는 모델 크기의 계속된 증가를 완화시켜줄 수 있는 방향이다.

TM은 source text와 그에 대응되는 번역들 페어를 데이터 베이스에 저장하는 방식이며 예전에는 SMT에 추가하여 많이 사용됐었다. 

최근에는 TM이 NMT 모델에도 마찬가지로 도움이 될 수 있음을 확인할 수 있었다. 이전 연구들과 마찬가지로 TM-augmented NMT 모델들은 학습 코퍼스를 학습 후에 버리지 않으며 실제 inference time에도 활용하는 방식으로 연구됐다. 활용하는 방법은 아래의 두 단계를 통해 볼 수 있다.

1. **Retrieval stage**
    1. 학습 corpus로부터 입력된 source 문장과 유사한 source text를 lexical overlap 기반으로 검색하는 방식(Nearest neighbor).
    2. Embedding based match 방식.
    3. Hybrid 방식.
2. **Generation stage**
    1. 검색된 번역 문장들을 메모리 네트워크를 통해 처리하여 NMT 모델에 주입하는 방식.
    2. 검색된 번역 문장들을 source input에 그대로 붙이는 방식.
    3. Decoding 할 때 단어 분포를 바이어스 시키는 방식(biasing the word distribution)
    4. knn-mt와 같이 token-level로 nearest neighbor search를 하는 방식

이런 차이들이 있음에도 불구하고 논문의 저자들은 **2가지 주요 문제점**을 찾았다.

1. **Translation memory가 Bilingual corpus여야만 하는 점**
    1. Bilingual corpus로 제한된 순간 수많은 monolingual corpus를 사용할 수 없게된다.
2. **Memory retriever은 non-learnable하며 end-to-end optimizing 그리고 특정 downstream NMT 모델에 적용이 힘든 점**
    1. 현재의 검색 방식은 BM25 모델을 사용하고 있다. 이 모델은 source 문장이 더 많이 겹칠 수록 대응되는 번역 문장 또는 부분들이 최종 번역 문장에 반영될 확률이 커지게 된다. 이런 방식이 맞을 수 있지만 가장 유사한 source 문장이 가장 정확한 번역 문장을 갖고있지 않기 때문에 문제가 될 수 있다.
    2. 논문의 저자들은 검색 방식이 task-dependent 방식으로 학습되길 원하며 이렇게 함으로써 정말로 memory가 최종 번역 품질에 도움이 될 때만 활용될 수 있길 바란다.

이 논문에서는 다음의 항목들을 제안한다.

1. augment NMT models with monolingual TM
2. a learnable cross-lingual memory retriever

특히, 저자들은 source-side 문장들과 그에 대응되는 target-side 번역들을 latent vector space로 align한다. Latent vector space로 align하는 방법은 dual-encoder framework을 이용하며 이 인코더를 이용해 생성된 distance는 검색을 위한 score function으로 사용된다.

그 결과로 논문에서 제안하는 memory retriever은 source-side input과 target-side translation을 연결할 수 있고 이를 통해 target 언어로된 monolingual data 자체로 TM으로 사용될 수 있다.

Translation을 실행하기 전 memory retriever은 monolingual 문장들과 target-side 학습 코퍼스(TM)으로부터 가장 높은 점수들을 선택합니다. 그 이후에 downstream NMT 모델은 메모리들의 정보를 이용해 번역을 실행합니다.

제안 모델은 differentiable neural network로 memory retriever을 구성하였기 때문에 downstream NMT 모델과 memory retriever 네트워크를 통합하여 전체를 학습할 수 있도록 했습니다. Retrieval 점수들은 attention score들에 사용돼 제일 유용한 memory들을 활용하게 됩니다.

이러한 방식으로 memory retrieval은 ent-to-end optimization이 가능해집니다. Optimization을 위한 목적 함수는 다음을 학습하도록 합니다.

- A retrieval that improves the golden translation's likelihood is helpful & should be rewarded, while an uniformative retrieval should be penalized

논문에서 제안한 framework를 학습하기 위해서는 한가지 문제점이 존재하는데 그 문제점은 random init으로 학습을 시작했을 때 retrieved memory들은 input과 전혀 관계가 없는것처럼 보이게됩니다. 따라서, memory retriever은 downstream NMT 모델에 긍정적인 영향을 주지 못하게되며, 의미있는 gradient도 얻지 못하게됩니다. 이러한 결과로 NMT 모델은 모든 retrieved memory들을 무시하도록 학습될 수 있습니다.

이러한 **cold-start problem**을 회피하기 위해서 저자들은 2개의 cross-alignmnet task들을 통해 retrieval model을 **warm-start하는 방식을 제안합니다.**

실험 결과는 3가지 결과를 보여줍니다.

1. **non-TM baseline NMT model & strong TM-augmented baseline들의 성능을 넘는 결과를 보여줍니다.**
    1. 기존의 TM-based는 bilingual 데이터만 사용했지만 제안 모델은 target-side만 활용한 차이점이 있습니다.
2. **low-resource scenario들에서 번역 품질을 상당히 끌어올릴 수 있었다.**
3. **Strong cross-domain transferability by hot-swapping domain-specific monolingual memory**

## Related Work

### TM-augmented NMT

### Retrieval for Text Generation

### NMT using Monolingual Data

## Proposed Approach

이번장에서는 번역 task를 다음의 과정을 통해 소개한다.

1. Retrieve then generate
2. Model design for the cross-lingual memory retrieval model
3. Model design for the momry-augmented translation model
4. How to optimize the two components jointly using standard maximum likelihood training

## Overview
<figure>

제안한 모델은 두 단계로 분해할 수 있다 ⇒ 1) Retrieve 2) Generate

1. Translation Memory(TM)은 target language로 구성된 문장들(Z)이다.
2. source 문장 x가 입력되면, retrieval model은 가장 도움이 될만한 memory 문장들을 retrieval score(f(x, z))에 근거해 M개 추출한다.(M << |Z|)
3. Translation 모델은 검색된 set {(z, f(x,z)}과 입력 x를 사용하여 번역 문장 y를 생성한다.
    1. **p(y | x, z1, f(x, z1), x2, f(x, z2), ... , zM, f(x, zM))**
    2. 또한, 검색 점수 f(x,z)는 메모리 z1 ~ zM의 attention score와 같이 재사용된다.
4. 학습중에는 translation reference와의 likelihood을 maximizing하며 translation model & retrieval model 모두 성능을 향상시킨다.

## Retrieval Model

검색 모델(retrieval model)은 입력된 source sentence와 가장 유사한 문장들(from monolingual TM)을 선택하는 과정에 모든 책임이 있다. 즉, 수많은 target 문장 후보들과의 비교를 해야하는 큰 computational cost가 발생할 수 있음 알 수 있는데 논문에서는 **simple dual-encoder framework**(Bromley et al., 1993)을 이용하여 이 문제를 해결하고자 하였다.

Simple dual-encoder framework를 이용하여 가장 연관있는 문장들을 선택하는 문제를 Maximum Inner Product Search(MIPS)로 해결하였다.

### Relevance score f(x, z) 구하기

Relevance score를 구하기 위해 simple dual-encoder 구조를 사용한다고 앞서서 이야기했다. 논문에서는 다음과 같이 f()함수를 정의하였다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e4a78e0d-de3c-412e-b7a5-d6c1a7ec3288/Untitled.png)

- E_src, E_tgt는 각각 source sentence encoder, target sentence encoder를 지칭한다.
- 각 encoder는 x와 z를 d-dimensional vector로 매핑해주는 역할을 한다.
- Encoder는 Transformer로 구성됐으며 입력 문장에 [BOS] 토큰을 붙인 후 출력된 [BOS] 위치의 embedding을 output으로 사용하였다.
- Dimension을 줄이기 위해 encoder의 결과 벡터를 다시한번 linear transformation한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/97b96bb2-bac0-481d-ba54-23f93608ac47/Untitled.png)

- Normalized 벡터는 [-1, 1] 값게된다.

위와 같은 과정을 통해 점수를 얻게되며 실제로 사용할때는 FAISS와 같은 검색 라이브러리를 사용하여 미리 indexing을 하고 좀 더 쉽고 빠르게 가장 값이 높은 M개의 메모리들을 구할 수 있다.

## Translation Model

입력 source 문장 x, M개의 relevant TM(z)와 relevance score(f(x, z))가 주어졌을 때 번역 문장 y는 다음의 조건부 확률로써 정의될 수 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec264b8b-0881-4f9c-ac28-9c53bd02a58a/Untitled.png)

그렇다면 이 번역된 문장 y가 어떻게 나오는지 살펴본다.

### Transformer

- 전형적인 Enc-Dec NMT 모델(Transformer)을 base로 한다.
- source encoder transformer가 입력 source 문장 x를 dense vector로 바꿔준다.
- Decoder는 auto-regressive하게 번역 문장 y를 출력한다.
- 매 time step t마다 디코더는 기존에 생성된 출력 토큰들 y_1:t-1과 source encoder에서 나온 출력 벡터를 통해 hidden state h_t를 생성한다.
- h_t는 다음 토큰 확률로 변환되며 디코딩이 다음 스텝으로 넘어간다.

그렇다면 이제 extra memory input들과 기존 transformer 모델을 합쳐야하므로 기존 enc-dec NMT 모델을 좀 더 확장해 변형해본다.

### Transformer + Memory

- 검색된 Translation memory들(Z)은 memory encoder를 통해 contextualized token embeddidng으로 변환된다.({z_i,k}, k=1 ~ L_i)
- 이때 변환된 contextualized token embedding의 dimension 크기는 z_i의 원본 길이(target sentence 길이)이다.
- 따라서, 변형된 토큰 임베딩의 길이가 문장 길이와 동일해지기 때문에 attention을 적용할 수 있게된다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/01bac0ea-7cd6-44cd-8d73-9527f127c9bf/Untitled.png)

- a_ij는 z_i의 j번째 토큰에 대한 attention score를 의미하며, c_t는 memory embedding의 weighted combination이다.(W_m, W_c는 학습 가능한 파라미터)
- 이렇게 생성된 attention 값을 cross attention이라하며 이 어텐션은 decoding에서 2번 사용된다.
    1. 디코더의 hidden state h_t가 weighted sum of memory embedding과 가중합돼 갱신된다. (⇒ h_t = h_t + c_t)
    2. 각각의 attention score는 토큰을 복사해 올 확률로써 다음 토큰 계산시에 사용된다.

    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51a59abb-1930-492d-9f4e-0f6f148c736c/Untitled.png)

    - 람다는 gating function으로써 feed-forward network로 구성되며 람다_t = g(h_t, c_t)로 계산된다.
- 최종적으로 translation output에서 retrieval model로 gradient upate을 진행해야하므로 attention score들을 relevance score들과 함께 사용하게 된다. 따라서 (1)번 식을 아래와 같이 재정의하게된다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a5420ff-93a6-423d-93e0-efc92e21782b/Untitled.png)

## Training

Retrieval 모델의 파라미터 세타와 번역 모델 파이를 sgd를 통해 업데이트를 진행한다. 위의 수식(2)에서 알 수 있듯이 reference 번역 문장으로 번역이 잘될만한 TM 문장들이 조금 더 높은 attention score와 더 높은 relevance socre을 얻게될 수 있도록 학습이 진행된다.

## Cross-alignment Pre-training

Retrieval 모델은 random initialization으로 시작한다. 하지만, 이렇게되면 모든 top TM 문장들은 입력 문장 x와 무관환 문장들이 검색될 것이다. 이런 무관한 문장들이 검색되면 retrieval model은 의미있는 gradient를 받을 수 없어 성능 향상에 도움이 되지 못하며 translation model은 TM input들을 무시하는 방향으로 학습하게 될 것이다. 이러한 cold-start 문제를 해결하기 위해 **two cross-alignment tasks를 통해 warm-start를 하는 방법을 제안한다.**

1. Sentence-level cross-alignmnet
    1. 이 task의 목적은 주어진 입력 source 문장에 대해 여러 번역 문장들 중 정확한 번역 문장을 찾도록 하는것이다.
    2. 매 학습 step마다 학습 코퍼스로부터 B개의 src-tgt 문장쌍들을 추출한다.
    3. 각 src 문장과 tgt 문장을 인코딩한 matrix X와 Z는 (B x d) 차원의 행렬이다.
    4. S = X * Z^t 를 계산해 (B x B) 차원의 relevance score 행렬을 얻을 수 있다.
    5. 각 src tgt문장들은 1 : 1 대응이므로 relevance score 행렬은 diagonal matrix 형태를 띄어야 가장 이상적일 것이다.
    6. 따라서 (X_i, Z_j) 가 i=j일때 align, 그렇지 않을땐 not align이 되도록 목적함수를 만들어 학습을 진행한다.

    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a722181-da6f-44e2-8a23-86f0730b1992/Untitled.png)

2. Token-level cross-alignment

    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/75e698c7-0898-48d6-86cd-12b803a9500b/Untitled.png)

    - 이 task의 목적은 source sentence representation이 들어왔을 때 알맞은 target language token이 예측될 수 있도록하는것이다.
    - Bag-of-words loss를 사용한다.
    - 확률값을 구하기 위해 linear projection + softmax function 형태의 layer를 사용한다

최종적으로 위의 두가지 Loss값을 joint하여 alignment task를 pretraining한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21222353-c47f-4993-946d-9420039efa00/Untitled.png)

## Asynchronous Index Refresh

MIPS를 빠르게 사용하기 위해서는 index에 있는 모든 memory들에 대해 encoding 값을 미리 계산해야만한다. 하지만, 학습중에는 retrieval 모델의 파라미터가 매번 바뀌기 때문에 고정될 수 없다. 2가지 해결방안이 있는데 다음과 같다.

1. Cross-alignment Pre-training 직후의 파라미터로 E_tgt 파라미터를 고정하고 E_src만 학습하는 방법
2. 비동기적으로 매번 index를 refresh하는 방법

# Experiments

논문에서는 3가지 환경하에 실험을 진행했다.(기존의 TM-augmented NMT 모델들은 첫번째 환경에서만 비교가 가능하다.)

1. Bilingual training corpus로만 TM을 제한.
2. Low-resource 환경(scarce bilingual, extra monolingual)
3. non-parametric domain adaptation using monolingual TM

## Implementation Details

- Transformer-base(8 attention head, 512 dim, 2048 feed-forward dim)
- Retrieval model - 3 block transformer
    - retrieve 5 top TM sentences
    - FAISS index code - IVF1024_HNSW32,SQ8
    - search depth - 64
- Memory encoder - 4 block transformer
- Translation model - 6 block transformer
- learning rate schedule, dropout and label smoothing
- Adam optimizer
- Train up to 100k steps
    - re-indexing interval - 3k training steps

## Conventional Experiments

### Data

- JRC-Acquis corpus
    - EN ↔ ES
        - Train : 679,088
        - Dev : 2,533
        - Test : 2,596
    - EN ↔ DE
        - Train : 699,569
        - Dev : 2,454
        - Test : 2,483

### Model

1. NMT without TM(Transformer-base)
2. TM-augmented NMT using source similarity search
    1. 기존에 사용되던 source 문장과 유사한 문장을 찾는 방식의 검색, fuzzy match system based on BM25 & edit distance
3. TM-augmented NMT using pre-trained cross-lingual retriever
    1. retrieval model을 cross-alignment tasks로 pretraining 시킨 후 retrieval 모델 파라미터를 고정한 모델
4. Proposed model using fixed TM index
5. Proposed model using asynchronous index refresh

### Results

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0afa5e39-1584-496f-8941-1d0916523334/Untitled.png)

1. Full model using asynchronous index refresh 모델이 가장 좋은 성능을 보였다. 이 결과로 monolingual TM이 NMT 성능을 올릴 수 있음을 확인 가능하다.
2. Model #3과 #4,5를 비교해보면 end-to-end 방식으로 retrieval 모델을 학습하는것이 중요한 성능 향상의 포인트인것을 알 수 있다.
3. Cross-lingual retrieval(model #4 & #5)는 source similarity search(model #2)보다 좋은 성능을 얻었다. Bilingual TM을 사용하는 model#2와 달리 monolingual TM만 필요로하는 model #4,5가 좋았음을 확인할 수 있다.

**Contrast to Previous Bilingual TM Systems**

기존 Bilingual TM 모델들과도 비교해보면 model#4, 5가 더 좋음을 확인할 수 있다. 이러한점은 논문에서 제안한 모델의 효과성을 입증하고 model#2가 기존 모델들보다 좋은것을 보아 번역 프레임워크 설계가 잘됐음도 확인할 수 있다.

## Low-Resource Scenarios

논문에서 제안한 모델의 특징은 monolingual TM만을 사용한다는점이다. 이런 특성으로 low-resource 상황에서의 실험 진행도 가능하다.

### Data

Training corpus를 4등분한 후 그중 1/4만 학습에 이용하고 나머지를 점차 monolingual TM으로 사용했을때와, 2/4만 사용했을때 두가지 환경으로 데이터를 구성했다.

### Models

Model #5가 가장 좋은 성능을 보였지만 학습 시간으로인해 model #4로 실험을 진행했다. 여기서 두가지 추가적인 환경을 구성했다.

1. 학습 pair들로만 TM을 제한하고 모델을 학습 & test time에 TM을 키워나가는 환경
    1. 학습하면서 한번도 보지 못한 학습 pair를 검색할 확률도 존재하게됨(?)
2. 학습때부터 TM을 키워나가며 re-train하는 환경

위의 두가지 환경과 Transformer base모델 그리고 model #2도 실험에 추가하여 평가하였다.

### Results

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6f26b649-bfc8-4211-a8a9-3d09134ad3bb/Untitled.png)

일반적인 경향성은 유사함을 확인할 수 있었다.

- TM이 커질수록 번역 품질이 좋아짐을 확인할 수 있다.
- 흥미롭게도 w/o re-training이 w/ re-training 모델과 비슷하거나 더 높은 경우도 있다는 점이다.
- 또한, 학습 코퍼스가 매우 적을때(1/4 bilingual) 적은양의 TM은 성능 하락의 요인이 될 수 있다는 점이다. 이러한 이유로는 아마 overfitting일 것으로 추정된다.

### Contrast to Back-Translation

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b897c226-197b-4e76-b1c0-08c7f5640796/Untitled.png)

Back-translation과의 비교 실험도 진행하였다. 이 방식도 monolingual을 활용하는 방식이므로 같이 실험을 진행하였다.

Table에서 볼 수 있듯이 1/4 학습셋만 이용했을때는 더 안좋은 성능이었지만 2/4 학습셋을 이용했을때부터 더 좋은 결과를 볼 수 있었다. 하지만, BT와 함께 논문에서 제안한 모델을 사용시 가장 좋은 결과를 얻을 수 있음을 볼 수 있다.

이러한 결과는 논문에서 제안하는 모델이 BT에 상호보완적인 작용을 해줄 수 있음을 알 수 있다.

## Non-parametric Domain Adaptation

Domain adaptation은 single general-domain model을 domain-specific monolingual TM을 활용하여 specific domain에 적용해보는 실험이다.

### Data

- German-English parallel data를 사용
- Medical, Law, IT, Koran, Subtitles 도메인이 존재
- 1/4 billingual pair만 학습에 사용하였으며 학습데이터의 target side 데이터 및 나머지 target side 데이터들을 domain specific monolingual TM을 구성하는데 사용
- dev, test로 2천문장을 사용하였다.

### Models

1. Transformer Base 모델을 모든 도메인들을 합쳐서 만든 코퍼스에 학습시켰다.
2. Model #4를 실험에 사용.

### Results

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/39194e47-f308-4cfd-95bb-0ebea4a87e15/Untitled.png)

1. Bilingual corpus만 사용 : 논문에서 제안한 모델이 더 적은 데이터를 사용했음에도 더 좋거나 약간 낮은 성능을 보임을 알 수 있었다.
2. Domain specific TM을 사용하면 훨씬 더 좋은 성능을 보임.
3. 모든 domain specific TM을 합쳐 사용했을때도 안사용했을때보다 좋은 성능을 보였음.

## Running Speed

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27d247f4-d983-43a3-b34d-7305332a39af/Untitled.png)

FAISS in-GPU index를 사용하여 실험을 진행하였다. 실험 결과 BM25기반의 검색보다 더 빠른 속도로 memory search가 가능했으며 학습과 테스트때의 속도차이는 위의 테이블로 확인 가능하다.

Memory-augmented model들이 기본적인 Transformer 모델보다 학습 convergence가 더 빠름도 관찰할 수 있었다고 한다.


## Terminology

### Translation Memory(TM)[[link](https://en.wikipedia.org/wiki/Translation_memory)]
Translation Memory란 "Segments"를 저장하는 데이터베이스이다. 이 Segments는 문장, 문단, 문장과 같은 단위(제목, 등)가 번역에 도움이되기 위해 기존에 번역됐던 유닛들이다. Translation memory는 source 문장과 그에 대응되는 target 문장을 저장한다.

### Siamese Network(Dual-Encoder)

### MIPS(Maximum Inner Product Search)
