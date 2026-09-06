# Brush Jam セットアップマニュアル

ローカルで動かすまでと、それを友人に公開するまでを、上から順に追えば済むように
書いた手順書です。英語の概要は [README.md](../README.md)、サーバーの詳細は
[PYTHON_SERVER.md](PYTHON_SERVER.md)、レンタル GPU への配備は
[RUNPOD_POD.md](RUNPOD_POD.md) にあります。

1. [必要なもの](#1-必要なもの)
2. [インストール](#2-インストール)
3. [モデルの用意](#3-モデルの用意)
4. [`.env` を書く](#4-env-を書く)
5. [起動と確認](#5-起動と確認)
6. [GPU が無い場合](#6-gpu-が無い場合)
7. [LAN 内で遊ぶ](#7-lan-内で遊ぶ)
8. [インターネットに公開する](#8-インターネットに公開する)
9. [運用上の注意](#9-運用上の注意)
10. [困ったとき](#10-困ったとき)

---

## 1. 必要なもの

| もの | バージョン | 用途 |
| --- | --- | --- |
| Python | 3.10 〜 3.12 | サーバー本体。3.13 は torch の都合で不可 |
| [uv](https://docs.astral.sh/uv/) | 最新 | Python の依存管理。`pip` は使いません |
| Node.js | 20 以上 | ブラウザクライアントのビルド。実行時には不要 |
| pnpm | 10 系 | `corepack enable` で入ります |
| NVIDIA GPU | VRAM 8 GB 以上 | 生成。無ければ mock で動きます(第 6 章) |
| SDXL チェックポイント | 6〜7 GB の `.safetensors` | 第 3 章でダウンロードします |

ディスクはモデル置き場に 10 GB、torch 込みの venv に 8 GB ほど見ておいてください。

Windows では Git Bash か PowerShell のどちらでも動きます。以下のコマンドは
どちらでもそのまま通るように書き、違うところだけ併記しています。

## 2. インストール

```bash
git clone https://github.com/tokoroten/brush_jam.git
cd brush_jam

pnpm install                 # クライアントの依存
pnpm build                   # クライアントをビルドして Python パッケージに入れる

cd apps/brushjam
uv sync --extra inproc       # torch + diffusers を含む venv を作る(数分かかります)
cd ../..
```

`pnpm build` は `vite build` の後に `scripts/build_web.py` が
`apps/brushjam/src/brushjam/static/` へコピーします。サーバーはこのディレクトリを
配信するので、**クライアントを直した後は必ず `pnpm build` を再実行**してください。
再実行しないと古いビルドが何のエラーも出さずに配信され続けます。

GPU を使わないなら `uv sync --extra inproc` の代わりに `uv sync` だけで済み、
torch が入らないぶん軽くなります。

## 3. モデルの用意

チェックポイントは 1 つで十分です。付属スクリプトが Civitai からダウンロードし、
`.env` に貼る行を表示します。

```bash
uv run --project apps/brushjam python apps/brushjam/scripts/download_models.py
```

- 保存先は `./models/checkpoints/`(gitignore 済み)。
- Civitai はほとんどのモデルでアカウントを要求します。401 / 403 が出たら
  Civitai の API キーを取得して `.env` の `CIVITAI_TOKEN` に書き、やり直してください。
  スクリプトは `.env` を自分で読むので、環境変数に export する必要はありません。
- 別のモデルを使いたいときは `CIVITAI_VERSION` にバージョン ID を書くか、
  手持ちの SDXL 系 `.safetensors` を `models/checkpoints/` に置いてパスを指定します。
  Illustrious 系で計測しているので、そこから離れると画風と最適な denoise が変わります。

4 ステップ用の DMD2 LoRA と fp16 版 VAE は初回起動時に Hugging Face から
自動ダウンロードされます(`./models/loras/`)。通常はトークン不要ですが、
レート制限にかかるときは `HF_TOKEN` に read 権限のトークンを入れてください。

## 4. `.env` を書く

```bash
cp .env.example .env
```

ローカル GPU で動かすなら書き換えるのは 1 行だけです。

```ini
INPROC_CHECKPOINT=./models/checkpoints/<ダウンロードしたファイル名>.safetensors
```

[`.env.example`](../.env.example) の他の値はすべてデフォルトそのもので、
コメントに意味が書いてあります。最初のうちに触りそうなのは次の程度です。

| 変数 | 既定 | いつ変えるか |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | 他の端末から繋がせるとき `0.0.0.0` |
| `PORT` | `8787` | ポートが被ったとき |
| `AI_BACKEND` | `inproc` | GPU が無ければ `mock` |
| `AI_PROFILE` | `fast` | 部屋の初期プロファイル。`quality` は数倍遅く数段よい |
| `PRESETS_R18` | `0` | R18 プリセット群をメニューに出すなら `1` |
| `HISTORY_DIR` | `./data/history` | 生成結果の保存先を変えるとき |
| `ROOM_CREATE_PER_MIN` | `10` | トンネル越しに公開するとき(第 8 章) |

`.env` は秘密情報を持つのでコミットしないでください(gitignore 済み)。
設定項目を増やしたら `.env.example` の側に書きます。

## 5. 起動と確認

```bash
pnpm start                   # uv run --project apps/brushjam brushjam
```

起動ログで、どのバックエンドが選ばれたかを確認します。

```
[brushjam] backend: inproc, <checkpoint>.safetensors resident in this process
[brushjam] server on http://127.0.0.1:8787
```

初回はモデルのロードとウォームアップに 30 秒〜1 分かかります。
<http://localhost:8787> を開いて名前を入れ、**Create room** で部屋を作り、
左に何か描いて、1〜2 秒後に右へ生成結果が出れば完了です。
部屋の URL `/r/<id>` を渡せば他の人も同じ部屋に入れます。

別ターミナルから、モデルを通した生成を 1 回だけ試す確認コマンドもあります。

```bash
pnpm smoke -- --url http://127.0.0.1:8787
pnpm latency -- --url http://127.0.0.1:8787 --n 5
```

開発中は `pnpm dev` を使うと、同じ Python サーバーの前に Vite の開発サーバーが
:5173 で立ち、クライアントの変更が即時反映されます。

## 6. GPU が無い場合

`.env` で `AI_BACKEND=mock` にすると、モデル無しで灰色の矩形が即座に返ってきます。
描画・同期・履歴・書き出しはすべて本物と同じ経路を通るので、
サーバーやクライアント側の作業はこれで足ります。

`AI_BACKEND=inproc` のまま torch もチェックポイントも無いと、
サーバーは黙って別のものに切り替えず起動を拒否します。
これは仕様で、`auto` だけが ComfyUI、mock の順にフォールバックします。

他の選択肢:

- **`comfyui`**: 起動中の ComfyUI(`COMFYUI_URL`)に投げる。同じ 8 GB カードに
  ComfyUI とこのサーバーの両方でモデルを載せると OOM するので、
  ComfyUI を使うならこちら一択です。
- **`stream`**: モデルを別マシンの [stream worker](STREAM_WORKER.md) に置く。
- **RunPod**: 第 8 章。

## 7. LAN 内で遊ぶ

全インターフェースにバインドして、自分のマシンの LAN アドレスを渡します。

```bash
HOST=0.0.0.0 pnpm start      # bash
pnpm dev:lan                 # 開発サーバー込みで同じことをする
```

```powershell
$env:HOST = '0.0.0.0'; pnpm start     # PowerShell
```

`.env` に `HOST=0.0.0.0` を書いておいても同じです。
相手には `http://<自分の LAN IP>:8787/r/<id>` を渡します。
Windows ファイアウォールが最初にダイアログを出したら「プライベートネットワーク」で
許可してください。出なかった場合は手動で 8787/TCP の受信を許可します。

```powershell
New-NetFirewallRule -DisplayName "Brush Jam" -Direction Inbound -Protocol TCP -LocalPort 8787 -Action Allow -Profile Private
```

## 8. インターネットに公開する

ローカルで立てたサーバーに外から繋がせる方法は主に 3 通り、
それに加えて自宅の GPU を使わない RunPod があります。
どれも **URL を知っている人は誰でも入れて、誰でも部屋を作れます**。
認証は一切ないので、URL が唯一のアクセス制御です(第 9 章)。

前提として、サーバーは 1 つの HTTP ポートしか使いません。
静的ファイル、`/api`、`/rooms`、そして `/ws/rooms/{id}` の WebSocket が
全部 :8787 に載っています。クライアントはページが https で開かれていれば
自動的に `wss://` で接続するので、HTTPS 終端をトンネル側に任せてもそのまま動きます。

### 8.1 トンネルを張る前の共通設定

`.env` を 1 か所変えます。

```ini
ROOM_CREATE_PER_MIN=60
```

理由: サーバーはレート制限に **接続元ソケットのアドレス**を使い、
`X-Forwarded-For` は読みません。トンネル経由だと全員が同じアドレス
(ngrok なら手元の ngrok エージェント、つまり 127.0.0.1)から来るように見えるので、
既定の毎分 10 部屋を全員で分け合うことになります。
RunPod の pod が 60 に上げているのも同じ理由です。

`HOST` は `127.0.0.1` のままで構いません。トンネルは手元のループバックに
繋ぎに来るので、LAN からの直接アクセスは閉じたままトンネルだけが入口になります。
LAN でも使うなら `0.0.0.0` に。

### 8.2 ngrok(いちばん手軽)

無料アカウントで足ります。<https://ngrok.com/> でサインアップし、
authtoken を 1 回だけ登録します。

```bash
# Windows: winget install ngrok.ngrok   /  macOS: brew install ngrok
ngrok config add-authtoken <あなたのトークン>
```

サーバーを起動したまま、別ターミナルで:

```bash
ngrok http 8787
```

表示される `https://xxxx-xx-xx.ngrok-free.app` の URL を相手に渡します。
部屋の URL は `https://xxxx.ngrok-free.app/r/<id>` になります。
WebSocket は追加設定なしで通ります。

知っておくこと:

- **無料枠の URL は起動のたびに変わります。** 固定したい場合は無料で 1 つ持てる
  static domain を使い、`ngrok http --url=<あなたの>.ngrok-free.app 8787` で張ります。
- 無料枠では初回アクセス時に ngrok の警告ページが 1 回挟まります。
  「Visit Site」を押せば以後は通ります。
- 無料枠には月間の転送量上限があります。生成結果は毎回 JPEG で全員に流れるので、
  長時間の部屋では消費が早いことを覚えておいてください。
- `ngrok http 8787 --basic-auth "user:password"` で簡易パスワードを掛けられます。
  部屋の URL しか渡さない前提より一段安全です。

### 8.3 Cloudflare Tunnel(固定 URL、無料)

自分のドメインを Cloudflare に置いているなら、こちらの方が固定 URL になって
警告ページも転送量上限もありません。ドメインが無くても、
ログイン無しの quick tunnel で ngrok と同じ使い捨て URL が得られます。

```bash
# Windows: winget install Cloudflare.cloudflared  /  macOS: brew install cloudflared

# 使い捨て URL(アカウント不要)
cloudflared tunnel --url http://127.0.0.1:8787
```

表示される `https://xxxx.trycloudflare.com` を渡します。

自分のドメインで固定するなら、1 回だけ:

```bash
cloudflared tunnel login
cloudflared tunnel create brushjam
cloudflared tunnel route dns brushjam jam.example.com
```

そして `~/.cloudflared/config.yml` に

```yaml
tunnel: brushjam
credentials-file: ~/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: jam.example.com
    service: http://127.0.0.1:8787
  - service: http_status:404
```

を書いて `cloudflared tunnel run brushjam` で起動します。
WebSocket は既定で通ります。Cloudflare Access を前に置けば
メールアドレスや Google アカウントでの認証も無料枠で付けられます。

### 8.4 Tailscale(身内だけ)

公開ではなく、決まったメンバーだけで遊ぶなら Tailscale が一番安全です。
全員が同じ tailnet に入り、`HOST=0.0.0.0` で起動して
`http://<あなたの tailscale 名>:8787/r/<id>` を渡すだけで、
インターネットには一切晒されません。
Tailscale Serve / Funnel を使えば https 化や一部公開もできます。

```bash
tailscale serve --bg 8787      # tailnet 内に https で配信
tailscale funnel --bg 8787     # インターネットにも公開
```

### 8.5 RunPod(自宅の GPU を使わない)

自宅の回線と GPU を使わず、レンタル GPU の上でサーバーごと動かす方法です。
`deploy/runpod/` がこのリポジトリを tarball にして pod へ送り、
起動まで面倒を見ます。SSH もレジストリも不要です。

```bash
# .env に RUNPOD_API_KEY, CIVITAI_TOKEN, HF_TOKEN を書いてから
uv run --project apps/brushjam python deploy/runpod/deploy.py deploy
uv run --project apps/brushjam python deploy/runpod/deploy.py status   # URL が出る
uv run --project apps/brushjam python deploy/runpod/deploy.py stop     # 課金を止める
```

RunPod のプロキシが HTTPS と WebSocket の面倒を見るので、そのまま公開 URL になります。
初回起動、再配備のコスト、放置時に自動停止させる `watch` など、詳細は
[RUNPOD_POD.md](RUNPOD_POD.md) を読んでください。

### どれを選ぶか

| 方法 | URL | 認証 | 向いている場面 |
| --- | --- | --- | --- |
| ngrok | 毎回変わる(固定も可) | basic auth | 今日この後 1 時間だけ遊ぶ |
| Cloudflare Tunnel | 固定(自分のドメイン) | Access で本格的に | 常設したい |
| Tailscale | tailnet 内 | メンバー限定 | 決まった仲間だけ |
| RunPod | pod ごとに固定 | なし | 自宅回線・GPU を使いたくない |

## 9. 運用上の注意

- **サーバーを再起動すると全部屋が消えます。** ストロークはメモリ上にしかなく、
  ディスクに残るのは生成結果の JPEG と JSON(`HISTORY_DIR`)だけです。
  誰かが描いているかもしれないときは確認してから再起動してください。
- **履歴はディスクに残ります。** 部屋ごと `HISTORY_ROOM_MB`(200)、全体で
  `HISTORY_TOTAL_MB`(2000)まで、古い順に消えます。公開するなら
  他人の描いたものが自分のディスクに残ることを承知の上で。
  `HISTORY_ENABLED=0` で完全に切れます。
- **プロンプトはサーバーで検査されません。** `PRESETS_R18` はメニューの
  表示だけを制御し、どんなプロンプトでも受け付けます。
  公開するなら誰に URL を渡すかで判断してください。
- **同時接続の上限** は部屋あたり `MAX_ROOM_SOCKETS`(16)、
  全体で `MAX_TOTAL_SOCKETS`(256)。GPU は 1 枚なので、
  部屋が増えても生成は全体で 1 つずつ順番に処理されます。
  部屋が 3 つ同時に動けば体感の待ち時間は 3 倍です。
- **ローカルの GPU は 1 モデルまで。** ComfyUI やベンチマークを同時に動かすと
  8 GB では OOM します。
- 公開中は `GET /healthz` で状態を見られます。`active_sockets` と
  `last_generation_at` が、今誰かいるかの判断材料です。

## 10. 困ったとき

**`inproc` にチェックポイントか torch が無いと言われて起動しない**
`.env` の `INPROC_CHECKPOINT` のパスが実在するか、
`uv sync --extra inproc` を `apps/brushjam` で実行したかを確認。
GPU 無しで動かしたいだけなら `AI_BACKEND=mock`。

**クライアントを直したのに画面が変わらない**
`pnpm build` を忘れています。サーバーは `static/` の最終ビルドを配信し続けます。

**CUDA out of memory**
他に GPU を使っているプロセス(ComfyUI、ブラウザの GPU 支援など)を止める。
それでも出るなら `INPROC_UNET_STORAGE=fp8` で 2.4 GB 空きます(1 回あたり
1 秒ほど遅くなります)。`quality` の 1024 は 8 GB で 7 GB 近くまで使います。

**生成が 6〜7 秒かかる(8 GB カード)**
`INPROC_VAE_TILE_SIZE` が `auto` になっているか確認。diffusers の既定 1024 では
768 も 1024 もタイル化されません。auto なら 12 GB 未満で 256 になります。

**トンネル経由で部屋が作れない(429)**
`ROOM_CREATE_PER_MIN` を上げてください(8.1 節)。全員が 1 アドレスに見えています。

**ngrok 越しに繋がるが生成結果が出ない、描いたものが同期しない**
ブラウザの開発者ツールで `/ws/rooms/` への接続が `wss://` になっているか、
101 で確立しているかを見る。`http://` の URL で開いていると混在コンテンツで
弾かれることがあるので、必ず `https://` の方を配ってください。

**Civitai から 401 / 403**
`CIVITAI_TOKEN` を `.env` に入れる。download_models.py は `.env` を自分で読みます。

**Node が古い**
`node --version` が 20 未満なら更新。ビルドにしか使わないので、
サーバーを動かすだけの環境には不要です。
