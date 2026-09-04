# 夜間作業レポート(2026-09-05 03:30 → 朝)

作業者: Fable(計画・検証・統合)、Opus ×2(実装 / ストリームワーカー)、Codex gpt-5.6-sol(レビュー)。
GPU は常に 1 モデルのみ常駐(ComfyUI とワーカーは排他)で運用した。

## 1. 結論(朝いちで知るべきこと)

- **推奨構成: ストリームワーカー + DMD2 LoRA + CFG 1.0、生成解像度 768、denoise 0.8。**
  実アプリ経由で 1 編集あたり約 1.4〜2.4 秒(デバウンス 0.4 秒込み)。ComfyUI 14 steps の 10.3 秒から約 5 倍高速。
- 起動方法(README 参照):
  ```
  cd apps/stream-worker && uv run stream-worker        # ワーカー(起動 ~40 秒、8GB で単独運用)
  pnpm dev:stream                                        # サーバ + Web(AI_BACKEND=stream)
  ```
  ComfyUI を同時に起動しないこと(VRAM スラッシングで 10 秒の処理が数分になる)。
- ComfyUI だけで使う場合は `pnpm dev`(fast プロファイル = DMD2 4 steps、768 で約 3.7 秒 / quality = 14 steps、1024 で約 10 秒)。

## 2. 実験結果の要点

### レイテンシ(RTX 3070 8GB、単独運用)

| 構成 | 768 | 1024 |
|---|---|---|
| ComfyUI 通常 14 steps | - | 10.3 s |
| ComfyUI fast(LCM 4 steps) | 3.7 s | 5.7 s |
| ワーカー LCM、旧 VAE | 6.0 s | 15.2 s |
| ワーカー LCM、fp16 VAE | 1.8 s | 3.5 s |
| ワーカー DMD2、CFG 1.0、fp16 VAE | **1.4 s** | - |

決定打はワーカー側の VAE: チェックポイントの VAE が fp32 に強制アップキャストされ、1024 のデコードだけで 9 秒かかっていた。fp16 修正版 VAE で 0.9 秒。
詳細: docs/experiments/2026-09-05-comfyui/REPORT.md、docs/experiments/2026-09-05-stream/REPORT.md、docs/STREAM_WORKER.md。

### 画質(denoise の意味)

- 0.5: ほぼ無変化。0.65: 装飾が付く(「描いた絵を整える」)。0.8: 再解釈が始まる(ノイズペンの空が建物列に)。0.9: 絵が消えてキャラ絵になる。
- 768 の方が 1024 より変換が大きく、かつ 2 倍速い。ノイズペンは 512 以下に縮小すると潰れる。
- LCM はノイズを「飾り」で埋め、DMD2 は「物」で埋める。DMD2 は CFG 1.0 でも劣化せず、LCM は CFG 1.0 で線が消える。
- CFG 1.0 ではネガティブプロンプトが効かない(UI で無効表示)。透かし風の文字は全構成で denoise 0.8 以上に出る(チェックポイント由来)。
- taesd VAE はさらに速い(1024 で 2.1 秒)が、階調部分がブロック状になるため既定にしない。

### StreamDiffusion 本体について

livepeer fork を実際にインストールして SDXL img2img を動かしたが、512 で 5.6〜6.3 秒(テキストエンコーダ常駐で VRAM 超過)、マスク入力なし、リクエスト単位の denoise なし、単一ファイル SDXL 読込にハック必要、のため不採用。upstream は SD1.5 専用。同じ思想(常駐 + 少ステップ + 埋め込みキャッシュ)を diffusers で自前実装したのが apps/stream-worker。

## 3. 実装した機能(夜間)

- ノイズペン(決定論的 hash ノイズ、ライブプレビュー、右端バグ修正)、ツール別ブラシサイズ(noise 既定 64 px)
- 1024² キャンバス + 全体再生成モード(既定)。大キャンバスの部分生成は `AI_MODE=patch` に温存
- 生成解像度をキャンバスと分離(Advanced で 512/768/1024)
- ルーム設定: denoise、ネガティブプロンプト、fast / quality プロファイル、バックエンド能力に応じた UI の出し分け
- 描画レイヤの Move(オフセット方式)、貼り付け画像の Move / 拡縮、レイヤパネルの整理、del / clear の確認ダイアログ
- ストリームワーカー(FastAPI + diffusers、/generate /healthz /cancel /load /unload、VAE 切替、キャンセル、出力検証)
- サーバ: `AI_BACKEND=stream`、ComfyUI 高速ワークフロー、レイテンシ計測・画質グリッドの各スクリプト
- Windows での開発サーバ再起動(EADDRINUSE)の修正

## 4. レビュー

Codex Sol を 7 ラウンド。第 5〜7 ラウンドの指摘(全体再生成のメモリ、スケジューラのデバウンス迂回、ステップ数の二重計算、縮小時のノイズ欠落、能力上限の未適用など)は全て Opus が修正。最終ラウンドの残件は本レポート末尾を参照。

## 5. 未解決・次にやること

- (朝に追記)
