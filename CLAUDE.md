# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

TorchFontは、ベクトルフォント向けのPyTorchディープラーニングライブラリ。Python (3.10+) と Rust (2024 edition) のハイブリッド構成で、PyO3/maturin を使用してビルドする。

主要機能:
- Google Fonts 統合（shallow Git clones + パターン認識フォント発見）
- Rust バックエンドで glyph outlines を PyTorch テンソルに直接レンダリング
- 合成可能な変換プリミティブ（truncation, batching, patch-based reshaping）

## 開発コマンド

### Python
```bash
uv run ruff format --diff   # フォーマットチェック
uv run ruff check           # Lint
uv run ty check             # 型チェック（mypy ではなく ty を使用）
uv run pytest tests         # テスト実行
uv run pytest tests -k "test_name"     # 単一テスト実行
uv run pytest tests --runslow          # 遅いテスト含む
uv run pytest tests --runnetwork       # ネットワークテスト含む
```

### Rust
```bash
cargo fmt --all                                      # フォーマット
cargo clippy --all-targets --all-features -D warnings # Lint
cargo check --all-targets --all-features             # チェック
```

### ビルド
```bash
maturin build --release  # wheel ビルド
maturin develop          # 開発用インストール
```

## アーキテクチャ

### Rust バックエンド (`src/`)
- `lib.rs` - PyO3 モジュール定義、`FontDataset` クラスを Python に公開
- `dataset/` - フォントデータセット処理（entry, index, io）
- `pen.rs` - glyph outline 描画の Pen 実装
- `error.rs` - エラー型定義

skrifa ライブラリでフォント処理、memmap2 でメモリ効率的なファイルアクセス。

### Python インターフェース (`torchfont/`)
- `datasets/` - PyTorch Dataset クラス
  - `FontFolder` - ローカルフォントディレクトリ
  - `FontRepo` - Git リポジトリベース
  - `GoogleFonts` - Google Fonts 統合
- `transforms/` - 変換パイプライン（Compose, LimitSequenceLength, Patchify）
- `io/` - Git 操作、アウトライン処理

### テスト (`tests/`)
- `conftest.py` で `--runslow` と `--runnetwork` マーカーを定義
- `tests/fonts/` にテスト用フォントファイルを配置

## 設定ノート

- ruff の lint は `select = ["ALL"]` で厳格、`EXE002` のみ無視
- ty の `invalid-method-override` は無視設定
- Rust edition は 2024、crate 名は `_torchfont`（Python から `_torchfont` としてインポート）
