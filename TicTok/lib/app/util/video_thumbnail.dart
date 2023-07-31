import 'package:path_provider/path_provider.dart';
import 'package:video_thumbnail/video_thumbnail.dart';

Future<String> createThumbnailFromVideoUrl(String videoUrl) async {
  final fileName = await VideoThumbnail.thumbnailFile(
    video: videoUrl,
    thumbnailPath: (await getTemporaryDirectory()).path,
    imageFormat: ImageFormat.PNG,
    quality: 0,
  );

  return fileName ?? '';
}
