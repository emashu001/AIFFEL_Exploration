import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Translate App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: TranslateScreen(),
    );
  }
}

class TranslateScreen extends StatefulWidget {
  @override
  _TranslateScreenState createState() => _TranslateScreenState();
}

class _TranslateScreenState extends State<TranslateScreen> {
  final TextEditingController _textEditingController = TextEditingController();
  String _translatedText = '';

  @override
  void dispose() {
    _textEditingController.dispose();
    super.dispose();
  }

  void _translateText() async {
    final userInput = _textEditingController.text;
    if (userInput.isNotEmpty) {
      final translation =
          await translateText(userInput, 'ko', 'en'); // 번역할 언어를 여기에 지정하세요.
      setState(() {
        _translatedText = translation;
      });
    }
  }

  Future<String> translateText(
      String text, String sourceLanguage, String targetLanguage) async {
    final apiUrl = Uri.parse('https://openapi.naver.com/v1/papago/n2mt');
    final headers = {
      'Content-Type': 'application/json; charset=UTF-8',
      'X-Naver-Client-Id': 'tRB21qsuuR_sG2A0mE1p',
      'X-Naver-Client-Secret': 'W97qLTe2sN',
    };

    final body = {
      'source': sourceLanguage,
      'target': targetLanguage,
      'text': text,
    };

    final response = await http.post(
      apiUrl,
      headers: headers,
      body: jsonEncode(body),
    );

    if (response.statusCode == 200) {
      final decodedResponse = jsonDecode(response.body);
      final translatedText =
          decodedResponse['message']['result']['translatedText'];
      return translatedText ?? '';
    } else {
      throw Exception('Failed to translate text');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Translate App'),
      ),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextField(
                controller: _textEditingController,
                decoration: InputDecoration(
                  hintText: 'Enter text to translate',
                ),
              ),
              SizedBox(height: 16.0),
              ElevatedButton(
                onPressed: _translateText,
                child: Text('Translate'),
              ),
              SizedBox(height: 16.0),
              Text(
                _translatedText,
                style: TextStyle(fontSize: 16.0),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
