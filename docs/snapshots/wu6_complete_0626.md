# WU-6 完成快照 — 写作定位/N4 reframe

> 完成 2026-06-26。修改对象：`docs/paper/paper.tex` + `docs/paper/references.bib`。
> 目标：把投稿稿从旧 n=60/单数据集/旧口径，重写为 **reading-not-reasoning + regime-dependent answer-source**。

## 完成项
- Related Work 扩到 WU-6 防线：
  - 威胁/近邻：2510.23482、2602.22766、CodeV 2511.19661。
  - 批评：2502.14829、2402.14897。
  - 谱系：Lanham、Turpin、Pfau、Lyu。
  - 内化：VPD、Zooming-without-Zooming、PEARL、ChartPaLI、Chart-R1、LOCUS。
  - 图表锚：ChartMuseum、PerceptionBottleneck、CharXiv、CHART-NOISe。
- Abstract 重写为 WU-5 headline 图主轴：
  - ChartQA/TabMWP accuracy gain 显著。
  - `F=flip_corrupt-flip_shuffle` 仍在 0 附近。
  - answer-time source = pixels / redundant text / conclusion copying。
- 加入 “Robustness to known confounds” 段：
  - 不 claim corrupt 干预本身新。
  - 用 shuffle/paraphrase/filler/truncate/delete/image-mask + snap/follow/other 化解 contextual-faithfulness 和 disguised-accuracy 风险。
- 更新功效句：
  - ChartQA n=400，MDE 约 8--9 points。
  - 8B/32B SFT discordant pairs 66/46；32B 37 gains vs 9 losses。
- 并入 WU-2/WU-3/WU-5/N1/N3 converging evidence：
  - ChartQA and TabMWP present-condition F<=0。
  - TabMWP gain subset: follow 1/40 and 0/57。
  - N3 natural counting follow 0/345 present, 1/345 masked。
  - N1 FinQA curriculum operand-follow 0/172 and 0/175。

## 红线检查
- Abstract 主句没有使用 “internalized reasoning is not load-bearing” 作为占位式主 claim。
- 正文明确写 “We do not claim the corrupt intervention itself is new”。
- 结论口径是：accuracy can move while load-bearing CoT faithfulness does not, in these regimes.

## 验证
- `python` 静态检查 LaTeX begin/end 与 brace depth：通过。
- citation key 检查：0 missing，26 cited keys。
- 旧数字残留检查：未发现 `.617/.733/.700/.767/p=.023/p=.125` 等旧 n=60 口径。
- `git diff --check docs/paper/paper.tex docs/paper/references.bib`：通过。
- PDF 已重编译：`conda run -n texlive tectonic paper.tex` → `docs/paper/paper.pdf`，11 页。
- 注意：`texlive` conda env 里的 `pdflatex/latexmk` 存在，但 `pdflatex.fmt` 未初始化且自动生成会找不到 `mktexlsr.pl`；本次使用 `tectonic` 成功编译。
