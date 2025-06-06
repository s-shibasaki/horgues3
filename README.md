# Horgues3 - 競馬予測システム

**🚧 開発中 / Under Development**

## 概要

Horgues3は、機械学習を活用した競馬の予測・馬券購入システムです。JV-Linkからのデータ収集、PyTorchによるディープラーニングモデルの学習、馬券確率計算まで一連の機能を提供します。

## 主な機能

- **データ収集**: JV-Linkを使用した競馬データの自動収集
- **機械学習**: 複数の損失関数（Plackett-Luce、RankNet等）を使用したランキング学習
- **馬券予測**: 単勝・複勝・馬連・馬単・ワイド・三連複・三連単の確率計算
- **自動購入**: KSC投票プラグイン2による自動馬券購入（開発予定）
- **資金管理・シミュレーション**: ケリー基準による賭け金計算とバックテスト機能（開発予定）
- **パドック映像分析**: コンピュータビジョンによる馬の状態分析（開発予定）
- **スケジュール実行**: 自動データ収集・予測・購入の定期実行（開発予定）

## プロジェクト構成

```
horgues3/
├── horgues3/           # メインPythonパッケージ
│   ├── __init__.py     # パッケージ初期化
│   ├── dataset.py      # データ処理・前処理
│   ├── models.py       # PyTorchモデル定義
│   └── betting.py      # 馬券確率計算
├── JVDataCollector/    # C# JV-Linkデータ収集ツール
│   ├── Program.cs      # メインプログラム
│   ├── JVLinkForm.cs   # GUI フォーム
│   └── ...             # その他のC#ファイル
├── notebooks/          # Jupyter Notebook（学習・分析）
│   └── train.ipynb     # モデル学習用ノートブック
├── scripts/            # 実行スクリプト
└── pyproject.toml      # Pythonプロジェクト設定
```

## セットアップ

### 必要な環境
- Python 3.11+
- PostgreSQL
- JRA-VAN Data Lab. SDK（データ収集に必要）

### インストール

1. リポジトリのクローン
```bash
git clone <repository-url>
cd horgues3
```

2. Python環境のセットアップ
```bash
pip install -e .
```

3. データベースの設定
```bash
# PostgreSQLの設定（詳細は別途ドキュメント参照）
```

### 使用方法

1. **データ収集**
   - Visual StudioでJVDataCollectorをビルド
   - JVDataCollectorを使用してJV-Linkからデータを収集
   - 処理の詳細は[JVLinkForm.cs](JVDataCollector/JVLinkForm.cs)を参照

2. **モデル学習**
   - [notebooks/train.ipynb](notebooks/train.ipynb)を使用してモデルを学習

3. **予測実行**
   - 予測プログラム（開発予定）

## 開発環境

### Python環境
- Python 3.11+
- PyTorch
- PostgreSQL
- Jupyter Notebook

### C#環境（JV-Linkデータ収集）
- .NET Framework 4.8+
- Visual Studio 2019/2022
- JRA-VAN Data Lab. SDK

### 外部サービス
- JRA-VAN Data Lab.（データ収集に必要な有料サービス）

## 開発状況

- [x] データ収集基盤
- [x] 機械学習モデル
- [x] 馬券確率計算
- [x] 予測性能評価
- [ ] 特徴量改良による性能改善
- [ ] モデル改良による性能改善
- [ ] 資金管理・バックテスト
- [ ] 自動購入
- [ ] スケジュール実行
- [ ] 地方競馬対応
- [ ] その他のデータソース
- [ ] パドック映像分析
- [ ] インストーラの作成

## ライセンス

このプロジェクトは開発中のため、ライセンスは未定です。

## 注意事項

- このシステムは投資助言を提供するものではありません
- 馬券購入は自己責任で行ってください
- JRA-VAN Data Lab.の契約が必要です
