# KelpJava
Kelp.Javaは、harujyouさん作の深層学習のライブラリKelpNetをJavaに移植したものです。

KelpNetはこちら https://github.com/harujoh/KelpNet

テスト1～13までの動作を確認することができます。

##KelpNetとの違い
APIは概ねKelpNetと同じ形で実装してあります。計算速度はKelpNetのリリースビルドと同程度です。
標準ライブラリ（例えばList等）名称はJavaのそれに置き換えてあります。

##セットアップ
1. KelpJavaをGithubからチェックアウトしてください。
2. KelpJavaはEclipseIDEで開発することができます。Eclipseのメニューから、File→Importを選択して、KelpJavaをワークスペースにインポートします。
2. テストプログラムは,src.testに配置してあります。Test1.javaを実行すると、XORの学習とテストが行われます。

##学習済みのデータセットについて
学習データセットはJavaのシリアライザで保存されるため、KelpNetと互換性がありません。また、今後のバージョンアップで学習データセットの互換性は維持されない可能性があります。



## License
Apache License 2.0
