import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('TensorFlow Lite Example')),
        body: Center(child: MyHomePage()),
      ),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String _prediction = '';

  @override
  void initState() {
    super.initState();
    predict();
  }

  Future<void> predict() async {
    final interpreter =
        await Interpreter.fromAsset('mobilenet_v1_1.0_224_quant.tflite');

    final imageBytes = await rootBundle.load('assets/image.jpg');
    final img.Image? decodedImage =
        img.decodeImage(imageBytes.buffer.asUint8List());

    if (decodedImage != null) {
      final resizedImage =
          img.copyResize(decodedImage, width: 224, height: 224);
      var input = _imageToByteListUint8(resizedImage);
      var output = Uint8List(1 * 1001);
      interpreter.run(input, output);

      List<dynamic> outputTensor = output.reshape([1, 1001]);
      final predictedLabel = _getPredictedLabel(outputTensor);

      setState(() {
        _prediction = 'Predicted label: $predictedLabel';
      });
    } else {
      setState(() {
        _prediction = 'Error: Could not decode the image';
      });
    }
  }

  Uint8List _imageToByteListUint8(img.Image image) {
    var convertedBytes = Uint8List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    for (var i = 0; i < 224; i++) {
      for (var j = 0; j < 224; j++) {
        var pixel = image.getPixel(j, i);
        convertedBytes[pixelIndex++] = img.getRed(pixel);
        convertedBytes[pixelIndex++] = img.getGreen(pixel);
        convertedBytes[pixelIndex++] = img.getBlue(pixel);
      }
    }
    return convertedBytes;
  }

  int _getPredictedLabel(List<dynamic> output) {
    double maxProb = 0;
    int maxIndex = 0;
    for (int i = 0; i < output[0].length; i++) {
      double prob = output[0][i].toDouble();
      if (prob > maxProb) {
        maxProb = prob;
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Text(
          'Prediction:',
          style: Theme.of(context).textTheme.headline5,
        ),
        Text(
          _prediction,
          style: Theme.of(context).textTheme.headline6,
        ),
      ],
    );
  }
}
