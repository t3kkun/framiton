# Framiton

日本手話の指文字動画を、元動画のフレーム番号に基づく時間区間として記録するWindows向けデスクトップアプリです。

## 使用技術

- Python 3.12+
- PySide6: デスクトップUI
- OpenCV: MP4などの動画フレーム読み込み
- FFprobe (FFmpeg): 利用可能な場合の可変フレームレート検出
- CSV: 研究用アノテーションの保存形式

## セットアップ

[uv](https://docs.astral.sh/uv/) とFFmpegをインストールしてください。FFmpeg/FFprobeはPATHから実行できるようにします。プロジェクトのフォルダーで以下を実行します。

```powershell
uv sync
```

uvがPython 3.12を管理して仮想環境を作成します。既存のPython環境へグローバルインストールしません。

## 起動

```powershell
uv run python main.py
```

「動画を開く」からMP4等を開くと、同じ名前のCSVが隣にある場合は自動で復元します。動画とCSVは別ファイルです。

## 操作

- Space: 再生 / 一時停止
- ← / →: 1フレーム戻る / 進む
- シークバー: 任意のフレームへ移動
- 「タグスタート」: 現在フレームを区間開始に記録
- 「タグゴール」: 終了位置を記録し、Transitionまたは指文字を選んで区間追加
- タイムライン上の色付き区間: クリックして種類、指文字、begin/endを編集または削除
- 保存 / 名前を付けて保存 / CSV読み込み: アノテーションファイルの管理

表示する現在フレーム番号とCSVのbegin/endは、どちらも0始まりの元動画フレーム番号です。時間表示はフレーム番号をメタデータのFPSで割った値です。VFRはFFprobeのフレーム時刻を確認し、ばらつきが検出された場合に警告します。

## CSVフォーマット

ヘッダーは `id,tag,character,begin,end` です。tagは `fingerspelling` または `transition`、Transitionのcharacterは空欄です。beginとendは両端を含むフレーム番号で、1レコードが1区間を表します。CSVから全区間を再読み込みでき、区間をフレームごとのデータへ展開する処理も後から追加できます。

```csv
id,tag,character,begin,end
1,fingerspelling,あ,18235,18280
2,transition,,18281,18300
```

CSVはExcelでも文字化けしにくいUTF-8 BOM付きで保存されます。CSV読み込み時には従来のUTF-8も読み込めます。

## 指文字リスト

選択候補は隣接する `characters.json` で管理します。ひらがな、カタカナ、小文字、濁点・半濁点、アルファベット、数字を含みます。候補は設定ファイルを編集すれば追加・変更できます。また、選択UIは新しい文字を直接入力することもできます。

## 今後の拡張

`VideoSource`は動画とフレーム取得だけを担当し、`Annotation`は時間区間データとして独立しています。将来Observer/MediaPipe/分類器からSuggestion（候補ラベルとconfidence）を提示しても、人間が確定したAnnotationを直接変更しない構造へ拡張できます。候補のAccept/Declineを区間編集フローに追加できます。
