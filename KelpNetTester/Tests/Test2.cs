﻿using System;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //MLPによるXORの学習【回帰版】
    class Test2
    {
        public static void Run()
        {
            //訓練回数
            const int learningCount = 10000;

            //訓練データ
            double[][] trainData =
            {
                new[] { 0.0, 0.0 },
                new[] { 1.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 1.0, 1.0 }
            };

            //訓練データラベル
            double[][] trainLabel =
            {
                new[] { 0.0 },
                new[] { 1.0 },
                new[] { 1.0 },
                new[] { 0.0 }
            };

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(2, 2),
                new ReLU(),
                new Linear(2, 1)
            );

            //optimizerを宣言(今回はAdam)
            nn.SetOptimizer(new Adam());


            //訓練ループ
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                //今回はロス関数にMeanSquaredErrorを使う
                nn.Train(trainData[0], trainLabel[0], LossFunctions.MeanSquaredError);
                nn.Train(trainData[1], trainLabel[1], LossFunctions.MeanSquaredError);
                nn.Train(trainData[2], trainLabel[2], LossFunctions.MeanSquaredError);
                nn.Train(trainData[3], trainLabel[3], LossFunctions.MeanSquaredError);

                //訓練後に毎回更新を実行しなければ、ミニバッチとして更新できる
                nn.Update(); 
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (var val in trainData)
            {
                var input = NdArray.FromArray(val);
                Console.WriteLine(input + ":" + nn.Predict(input));
            }
        }
    }
}