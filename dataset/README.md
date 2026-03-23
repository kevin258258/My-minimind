<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/jingyaogong/minimind/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/jingyaogong/minimind/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

</div>


# 📌 数据介绍

## Ⅰ Tokenizer

分词器将单词从自然语言通过“词典”映射到`0, 1, 36`这样的数字，可以理解为数字就代表了单词在“词典”中的页码。
可以选择自己构造词表训练一个“词典”，代码可见`./scripts/train_tokenizer.py`（仅供学习参考，若非必要无需再自行训练，MiniMind已自带tokenizer）。
或者选择比较出名的开源大模型分词器，
正如同直接用新华/牛津词典的优点是token编码压缩率很好，缺点是页数太多，动辄数十万个词汇短语；
自己训练的分词器，优点是词表长度和内容随意控制，缺点是压缩率很低（例如"hello"也许会被拆分为"h e l l o"
五个独立的token），且生僻词难以覆盖。
“词典”的选择固然很重要，LLM的输出本质上是SoftMax到词典N个词的多分类问题，然后通过“词典”解码到自然语言。
因为MiniMind体积需要严格控制，为了避免模型头重脚轻（词嵌入embedding层参数在LLM占比太高），所以词表长度短短益善。

<details style="color:rgb(128,128,128)">
<summary>Tokenizer介绍</summary>

第三方强大的开源模型例如Yi、qwen、chatglm、mistral、Llama3的tokenizer词表长度如下：

<table>
  <tr><th>Tokenizer模型</th><th>词表大小</th><th>来源</th></tr>
  <tr><td>yi tokenizer</td><td>64,000</td><td>01万物（中国）</td></tr>
  <tr><td>qwen2 tokenizer</td><td>151,643</td><td>阿里云（中国）</td></tr>
  <tr><td>glm tokenizer</td><td>151,329</td><td>智谱AI（中国）</td></tr>
  <tr><td>mistral tokenizer</td><td>32,000</td><td>Mistral AI（法国）</td></tr>
  <tr><td>llama3 tokenizer</td><td>128,000</td><td>Meta（美国）</td></tr>
  <tr><td>minimind tokenizer</td><td>6,400</td><td>自定义</td></tr>
</table>

> 👉2024-09-17更新：为了防止过去的版本歧义&控制体积，minimind所有模型均使用minimind_tokenizer分词，废弃所有mistral_tokenizer版本。

```
# 一些自言自语
> 尽管minimind_tokenizer长度很小，编解码效率弱于qwen2、glm等中文友好型分词器。
> 但minimind模型选择了自己训练的minimind_tokenizer作为分词器，以保持整体参数轻量，避免编码层和计算层占比失衡，头重脚轻，因为minimind的词表大小只有6400。
> 且minimind在实际测试中没有出现过生僻词汇解码失败的情况，效果良好。
> 由于自定义词表压缩长度到6400，使得LLM总参数量最低只有25.8M。
> 训练数据`tokenizer_train.jsonl`均来自于`匠数大模型数据集`，这部分数据相对次要，如需训练可以自由选择。
```

</details>

## Ⅱ Pretrain数据

经历了MiniMind-V1的低质量预训练数据，导致模型胡言乱语的教训，`2025-02-05` 之后决定不再采用大规模无监督的数据集做预训练。
进而尝试把[匠数大模型数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)的中文部分提取出来，
清洗出字符`<512`长度的大约1.6GB的语料直接拼接成预训练数据 `pretrain_hq.jsonl`，hq即为high
quality（当然也还不算high，提升数据质量无止尽）。

文件`pretrain_hq.jsonl` 数据格式为

```bash
{"text": "如何才能摆脱拖延症？ 治愈拖延症并不容易，但以下建议可能有所帮助..."}
```

## Ⅲ SFT数据

[匠数大模型SFT数据集](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
“是一个完整、格式统一、安全的大模型训练和研究资源。
从网络上的公开数据源收集并整理了大量开源数据集，对其进行了格式统一，数据清洗，
包含10M条数据的中文数据集和包含2M条数据的英文数据集。”
以上是官方介绍，下载文件后的数据总量大约在4B tokens，肯定是适合作为中文大语言模型的SFT数据的。
但是官方提供的数据格式很乱，全部用来sft代价太大。
我将把官方数据集进行了二次清洗，把含有符号污染和噪声的条目去除；另外依然只保留了总长度`<512`
的内容，此阶段希望通过大量对话补充预训练阶段欠缺的知识。
导出文件为`sft_512.jsonl`(~7.5GB)。

[Magpie-SFT数据集](https://www.modelscope.cn/organization/Magpie-Align)
收集了~1M条来自Qwen2/2.5的高质量对话，我将这部分数据进一步清洗，把总长度`<2048`的部分导出为`sft_2048.jsonl`(~9GB)。
长度`<1024`的部分导出为`sft_1024.jsonl`(~5.5GB)，用大模型对话数据直接进行sft就属于“黑盒蒸馏”的范畴。

进一步清洗前两步sft的数据（只保留中文字符占比高的内容），筛选长度`<512`的对话，得到`sft_mini_512.jsonl`(~1.2GB)。

所有sft文件 `sft_X.jsonl` 数据格式均为

```text
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```

## Ⅳ RLHF数据

来自[Magpie-DPO数据集](https://www.modelscope.cn/datasets/Magpie-Align/MagpieLM-DPO-Data-v0.1)
大约200k条偏好数据（均是英文）生成自Llama3.1-70B/8B，可以用于训练奖励模型，优化模型回复质量，使其更加符合人类偏好。
这里将数据总长度`<3000`的内容重组为`dpo.jsonl`(~0.9GB)，包含`chosen`和`rejected`两个字段，`chosen`
为偏好的回复，`rejected`为拒绝的回复。

文件 `dpo.jsonl` 数据格式为

```text
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

## Ⅴ Reason数据集：

不得不说2025年2月谁能火的过DeepSeek...
也激发了我对RL引导的推理模型的浓厚兴趣，目前已经用Qwen2.5复现了R1-Zero。
如果有时间+效果work（但99%基模能力不足）我会在之后更新MiniMind基于RL训练的推理模型而不是蒸馏模型。
时间有限，最快的低成本方案依然是直接蒸馏（黑盒方式）。
耐不住R1太火，短短几天就已经存在一些R1的蒸馏数据集[R1-Llama-70B](https://www.modelscope.cn/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)、[R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT)、
[Alpaca-Distill-R1](https://huggingface.co/datasets/shareAI/Alpaca-Distill-R1-ZH)、
[deepseek_r1_zh](https://huggingface.co/datasets/jinliuxi/deepseek_r1_zh)等等，纯中文的数据可能比较少。
最终整合它们，导出文件为`r1_mix_1024.jsonl`，数据格式和`sft_X.jsonl`一致。

## Ⅵ 更多数据集

目前已经有[HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)
在收集和梳理中文LLM相关的开源模型、应用、数据集及教程等资料，并持续更新这方面的最新进展。全面且专业，Respect！

---

## Ⅷ 数据集下载

> [!NOTE]
> 2025-02-05后，开源MiniMind最终训练所用的所有数据集，因此无需再自行预处理大规模数据集，避免重复性的数据处理工作。

MiniMind训练数据集 ([ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong))

> 无需全部clone，可单独下载所需的文件

将下载的数据集文件放到`./dataset/`目录下（✨为推荐的必须项）

```bash
./dataset/
├── dpo.jsonl (55MB)
├── lora_identity.jsonl (22.8KB)
├── lora_medical.jsonl (34MB)
├── pretrain_hq.jsonl (1.6GB, ✨)
├── r1_mix_1024.jsonl (340MB)
├── sft_1024.jsonl (5.6GB)
├── sft_2048.jsonl (9GB)
├── sft_512.jsonl (7.5GB)
└── sft_mini_512.jsonl (1.2GB, ✨)
```

<details style="color:rgb(128,128,128)">
<summary>注：各数据集简介</summary>

* `dpo.jsonl` --RLHF阶段数据集
* `lora_identity.jsonl` --自我认知数据集（例如：你是谁？我是minimind...），推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `lora_medical.jsonl` --医疗问答数据集，推荐用于lora训练（亦可用于全参SFT，勿被名字局限）
* `pretrain_hq.jsonl`✨ --预训练数据集，整合自jiangshu科技
* `r1_mix_1024.jsonl` --DeepSeek-R1-1.5B蒸馏数据，每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_1024.jsonl` --整合自Qwen2.5蒸馏数据（是sft_2048的子集），每条数据字符最大长度为1024（因此训练时设置max_seq_len=1024）
* `sft_2048.jsonl` --整合自Qwen2.5蒸馏数据，每条数据字符最大长度为2048（因此训练时设置max_seq_len=2048）
* `sft_512.jsonl` --整合自匠数科技SFT数据，每条数据字符最大长度为512（因此训练时设置max_seq_len=512）
* `sft_mini_512.jsonl`✨ --极简整合自匠数科技SFT数据+Qwen2.5蒸馏数据（用于快速训练Zero模型），每条数据字符最大长度为512（因此训练时设置max_seq_len=512）

</details>


![dataset](./images/dataset.jpg)

<details style="color:rgb(128,128,128)">
<summary>说明 & 推荐训练方案</summary>

* MiniMind2 Series均经过共约20GB语料训练，大约4B tokens，即对应上面的数据组合训练结果（开销：💰💰💰💰💰💰💰💰，效果：😊😊😊😊😊😊）

* 想要最快速度从0实现Zero模型，推荐使用`pretrain_hq.jsonl` + `sft_mini_512.jsonl` 的数据组合，具体花销和效果可查看下文表格（开销：💰，效果：😊😊）

* 推荐具备一定算力资源或更在意效果的朋友可以考虑前者完整复现MiniMind2；仅有单卡GPU或在乎短时间快速复现的朋友强烈推荐后者；

* 【折中方案】亦可选择例如`sft_mini_512.jsonl`、`sft_1024.jsonl`中等规模数据进行自由组合训练（开销：💰💰💰，效果：😊😊😊😊）。

</details>