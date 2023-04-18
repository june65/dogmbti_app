import 'dart:typed_data';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:dogmbti/firebase_options.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

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
    final interpreter = await Interpreter.fromAsset('dog_not_dog_v3.tflite');

    final imageBytes = await rootBundle.load('assets/image_cat.jpg');
    final img.Image? decodedImage =
        img.decodeImage(imageBytes.buffer.asUint8List());

    if (decodedImage != null) {
      final resizedImage =
          img.copyResize(decodedImage, width: 224, height: 224);

      print(interpreter.getInputTensors()[0]);
      print(interpreter.getOutputTensors()[0]);
      var input = _imageToByteListFloat32(resizedImage);
      var output = Float32List(5).reshape([1, 5]);
      interpreter.run(input, output);
      print(output);
      List<dynamic> outputTensor = output;
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

  /*
  Float32List _imageToByteListFloat32(img.Image image) {
    var convertedBytes = Float32List(1 * 224 * 224 * 3);
    int pixelIndex = 0;
    for (var i = 0; i < 224; i++) {
      for (var j = 0; j < 224; j++) {
        var pixel = image.getPixel(j, i);
        convertedBytes[pixelIndex++] = img.getRed(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getGreen(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getBlue(pixel) / 255.0;
      }
    }
    return convertedBytes;
  }
  */

  List<List<List<List<double>>>> _imageToByteListFloat32(img.Image image) {
    var convertedBytes = List.generate(
      1,
      (index) => List.generate(
        224,
        (index) => List.generate(
          224,
          (index) => List.generate(
            3,
            (index) => 0.0,
          ),
        ),
      ),
    );

    for (var i = 0; i < 224; i++) {
      for (var j = 0; j < 224; j++) {
        var pixel = image.getPixel(j, i);
        convertedBytes[0][i][j][0] = img.getRed(pixel) / 255.0;
        convertedBytes[0][i][j][1] = img.getGreen(pixel) / 255.0;
        convertedBytes[0][i][j][2] = img.getBlue(pixel) / 255.0;
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
        GestureDetector(
          onTap: () async {
            await FirebaseFirestore.instance
                .collection('result')
                .doc('data')
                .set({
              'result': 1,
            });
          },
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Icon(Icons.camera),
              SizedBox(
                width: 10,
              ),
              Text('카메라'),
            ],
          ),
        )
      ],
    );
  }
}
