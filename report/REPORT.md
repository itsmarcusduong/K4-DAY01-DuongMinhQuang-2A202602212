# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): class_id = 468, class_name = cab, rank = 1, score = 0.510915, taxonomy_name = ImageNet-1K
- Record này mô tả toàn ảnh như thế nào? Ảnh là cảnh giao thông đông đúc với nhiều xe buýt và ô tô. cab mô tả toàn ảnh theo tín hiệu nổi bật là phương tiện đường bộ; nó không có nghĩa mọi vật thể đều là taxi và không chứng minh đây là nhãn đúng. Top 5 là cab, minibus, police_van, recreational_vehicle, streetcar. Các lớp đều liên quan giao thông nên tương đối phù hợp, nhưng minibus có vẻ trực quan hơn cab vì nhiều xe buýt xuất hiện rõ, còn taxi không nổi trội
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Taxonomy của dữ liệu huấn luyện checkpoint quyết định; ở đây là ImageNet-1K. Người chạy inference không tự thêm lớp nếu không đổi hoặc huấn luyện checkpoint.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? class_id là khóa máy đọc trong một taxonomy; class_name giúp con người hiểu; taxonomy_name xác định không gian nhãn chứa ID. Nhờ vậy tránh nhầm ID giữa taxonomy và tránh vấn đề tên trùng/đổi tên
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Phải nêu rõ single-label hay multi-label; tiêu chí chọn nhãn chính (độ nổi bật, diện tích, mục đích ảnh hay ngữ cảnh); có gán chủ thể phụ k
- Vì sao model score không phải ground truth? Score chỉ là độ tin cậy/ưu tiên nội bộ của model. Nó không thay thế nhãn do người tạo hoặc xác nhận theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name = person, score = 0.912625, bbox_xyxy = [385.33, 69.24, 498.92, 348.92], bbox_width = 113.58 px, bbox_height = 279.68 px
- Diễn giải vị trí box bằng lời: Với gốc (0, 0) ở góc trên trái, hộp chạy từ khoảng x=385 đến 499 và y=69 đến 349, bao quanh gần toàn bộ phần người nhìn thấy từ đầu đến chân. Hộp khá chặt và chỉ chứa ít nền quanh cơ thể.
- So sánh số prediction ở hai threshold: kitchen: 0.20 cho 17 prediction; 0.35 cho 11; 0.60 cho 6. Tăng từ 0.20 lên 0.60 lọc 11 prediction score thấp, gồm các lớp như cup, spoon, potted plant, dining table, bottle.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp tăng khả năng bao phủ vật thể khó nhưng làm reviewer phải xem nhiều prediction và có thể tăng false positive. Threshold cao giảm khối lượng review nhưng tăng nguy cơ bỏ sót. Đây chỉ là bộ lọc prediction, không phải quy tắc xóa ground truth.
- Đề xuất một quy tắc box chặt: Dùng hình chữ nhật nhỏ nhất bao hết phần vật thể nhìn thấy rõ; không cắt pixel rõ ràng của vật thể và không thêm nền nếu tránh được. Guideline nên định lượng dung sai biên.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline phải chọn bao phần nhìn thấy hay ước lượng toàn vật thể. Đề xuất chỉ bao phần nhìn thấy, cho hộp chạm mép ảnh khi bị cắt và gắn cờ occluded/truncated nếu schema hỗ trợ. Trường hợp không đủ đặc trưng để xác định lớp hoặc tách vật thể phải escalation, không tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id = kitchen-001, class_name = person, score = 0.899318, polygon có 348 điểm. Đa giác đi từ đỉnh đầu, theo sát tóc, vai, tay, thân áo/tạp dề và tách theo hai chân trước khi khép quanh phần người nhìn thấy
- Polygon bổ sung chi tiết gì so với box? Box [385.45, 66.44, 498.02, 348.58] chỉ cho hình chữ nhật bao ngoài. Polygon bổ sung đường cong đầu/vai, phần lõm giữa tay và thân, khoảng trống giữa hai chân, đồng thời loại nền ở các góc hộp
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id chỉ phân biệt các đối tượng riêng trong output; không phải class_id, không cho biết loại đối tượng và không phải tracking ID bền vững giữa ảnh/frame hay lần chạy
- Đề xuất một quy tắc biên mask: Theo ranh giới pixel nhìn thấy của từng instance, lấy đủ phần vật thể rõ ràng và loại nền/vật thể che phía trước; không tự vẽ phần bị che nếu dự án dùng visible-mask
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định mức zoom, cách đặt biên mờ, bóng/bán trong suốt có được tính không, cách tách instance tiếp xúc và dùng visible-mask hay amodal-mask. Nếu không thể xác định nhất quán, đánh dấu ambiguous và escalation

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh theo taxonomy | Nhiều loại xe; top-1 cab chưa chắc là chủ thể chính | Xem toàn ảnh, áp dụng quy tắc single/multi-label | Đúng taxonomy, tiêu chí chủ thể chính và tính nhất quán |
| Phát hiện vật thể | Mỗi object: lớp + box xyxy pixel | Threshold cao có thể bỏ sót; box có thể thừa nền/cắt vật thể | Tạo/sửa box chặt cho mọi object thuộc phạm vi | Đúng lớp, đủ object, box chặt, cờ che/cắt đúng |
| Instance segmentation | Mỗi instance: lớp + polygon/mask | Biên phức tạp, tiếp xúc và che khuất | Vẽ/sửa mask theo biên và tách từng instance | Không tràn nền/mất phần rõ; tách và xử lý che nhất quán |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng ảnh đúng phạm vi và quyền truy cập; không sao chép, chia sẻ hoặc đưa dữ liệu nhạy cảm/PII vào báo cáo hay output công khai
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: giảng viên/người phụ trách dữ liệu hoặc đầu mối escalation trong guideline

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
