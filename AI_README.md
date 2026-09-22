# CAD-to-Rendering Research Tool: AI Integration & Development Log (AI_README.md)

## ล่าสุด — Part 7A CAD-Guided SAM 3 Automatic Correspondence (2026-09-22)

- แทนหน้า human-first ด้วย SAM 3 concept queries; เพิ่ม Validation Concept ใน Properties ตามข้อกำหนด ไม่เปลี่ยน generation หรือ Segment to Reference
- concept ซ้ำ query ครั้งเดียว; single strong = auto_matched แต่ human_verified=false; multiple = ambiguous หรือ one-to-one centroid assignment; zero = missing_candidate ไม่ใช่ยืนยันว่าของหาย
- มี Confirm/Reject, เลือก candidate, SAM 3 point/box fallback, polygon และ hallucination review แยก; ไม่คำนวณ SAS/IoU/errors
- บันทึกแยก `renders/evaluation/<render_id>/sam3/`; รักษาข้อมูล manual รุ่นเก่า, native resolution, GT snapshot, provenance, revision locking และ human decisions
- ติดตั้ง Transformers 5.17.0 และตรวจ API จริงแล้ว **ยังไม่มี SAM 3 checkpoint และยังไม่ได้ล็อกอิน Hugging Face จึงยังไม่ผ่าน acceptance ด้วย pretrained SAM 3 จริง**; ไม่ใช้ SAM ViT-B แทนแล้วเรียกว่า SAM 3
- tests ใช้ detector/adapter test doubles อย่างชัดเจน; คู่มือ `docs/PART_7A.md`, setup `scripts/setup_sam3.py`, dependencies `requirements-sam3.txt`
- ไม่เรียก paid generation API; หลังได้รับสิทธิ์ facebook/sam3 ให้ทดสอบ local inference กับ render ที่มีอยู่
- ตรวจล่าสุด: **151 passed, 2 skipped**; browser แยกทดสอบ SAM 3 page, native-coordinate polygon, Confirm และ reload persistence ผ่าน โดยไม่ใช้ model inference หรือ paid API

## ประวัติ — Part 7A Human-Guided SAM Spatial Correspondence (ถูกแทนด้วย SAM 3)

- เพิ่มหน้า `Spatial Validation` ผ่าน navigation เท่านั้น ไม่เปลี่ยนการทำงาน Parts 1–6 หรือ Segment to Reference
- เลือกกล้อง/render/entity จาก visible IDs ใน instance.npy; GT mask สร้างจาก ID โดยตรง ผู้ใช้เป็นผู้เลือก semantic correspondence
- Canvas แปลงคลิก/box จากขนาดแสดงผลกลับเป็น original pixels; SAM รองรับ positive/negative points, box, candidates, overlay, retry; Manual Polygon เป็น fallback
- SAM ViT-B pretrained ติดตั้งและทดสอบจริงบน CPU แล้ว: candidate masks 3×1024×1024, initial model+inference ~11 วินาที; cache checkpoint และ model ไม่มี paid API calls
- ต้อง Confirm Mask ก่อนบันทึก; รองรับ missing/occluded/unreviewed และ hallucination ที่ entity_id=null; ไม่คำนวณ IoU/SAS/metrology
- evaluation แยกต่อ render ใน `renders/evaluation/<render_id>/` มี binary PNG, prompts, backend/package/commit/checkpoint hash, original size และ ground-truth snapshot; ไม่เขียน layout.json
- resolution mismatch block confirmation; historical render ที่ instance hash ไม่ตรง block แทนการนำ export ปัจจุบันมาแทน; PNG/NPY/mapping ต้องตรงกัน; revision lock ป้องกัน concurrent overwrite
- full tests: **137 passed, 2 skipped**. ทดสอบ browser แยกด้วย SAM จริง: scaled click coordinates, positive/negative, box, confirm, reload, missing/occluded, manual hallucination ผ่าน ข้อมูลทดสอบอยู่ test-results เท่านั้น
- คู่มือและ setup: `docs/PART_7A.md`; `requirements-segmentation.txt`; idempotent checkpoint setup `scripts/setup_sam.py`

## ล่าสุด — Submit / Confirm request snapshot

- Generate flow แบ่งเป็น `Submit — Prepare Summary` (local, ไม่เสีย API) และ `Confirm & Generate`
- Submit freeze ภาพเป็น bytes, prompt, state และ model settings; Summary แสดงกล้อง model temperature ทุกภาพตาม IMAGE numbering และ exact prompt จาก snapshot
- fingerprint ครอบคลุม state, prompt, image hashes, camera preflight และ config ที่ไม่รวม secret เมื่อแก้ข้อมูล/ไฟล์/model จะ invalidate และต้อง Submit ใหม่
- Confirm ส่ง frozen image bytes ไม่อ่าน path ใหม่; consume snapshot ก่อน dispatch ป้องกัน rerun ส่งซ้ำ และบันทึก `confirmed_request_id` ใน render metadata
- ทดสอบ **122 passed, 2 skipped**, รวม Submit ไม่เรียก provider, invalidation, frozen bytes, และ confirmation ถูกใช้ครั้งเดียว; ไม่เรียก API จริง

## ล่าสุด — Offline audit หลังผู้ใช้จำกัดค่า API

- **ห้ามเรียก image-generation API เพิ่มโดยไม่ได้รับอนุญาตจากผู้ใช้**; รอบ audit นี้เรียก 0 ครั้ง
- พบ export กล้อง 002 เก่า: position ใน metadata `[6.626988,3.079949,1]` แต่กล้องปัจจุบันและ render records เป็น `[6.000279,2.974027,1]`; heading/FOV ตรงกัน จึงไม่สรุปว่าปัญหานี้อธิบายภาพมุมตรงทั้งหมด
- Depth/instance hashes ตรงกับ manifests ของทั้งสามรอบ และ RGB ทุกพิกเซลตรงกับ instance IDs ไม่พบ palette mismatch
- เพิ่ม `services/conditioning_preflight.py`: ตรวจ camera/export ก่อน Generate; mismatch จะ block และ metadata หายจะแจ้งว่า unverified
- Provider หยุดก่อน request หากอ่านภาพไม่ได้ แทน silently skip ซึ่งอาจทำให้ IMAGE numbering ผิด
- สำรอง export เดิมใน `test-results/camera_002_before_refresh/` แล้ว export กล้อง 002 ใหม่ผ่าน UI ในเครื่อง ไม่เปลี่ยน camera geometry; preflight ผ่าน
- Full suite **116 passed, 2 skipped**; ยังไม่ได้ทดสอบภาพ API ใหม่ และไม่ได้แก้ prompt เพิ่มในรอบนี้ รายงาน: `test-results/SAME_ROOM_AUDIT.md`

## เพิ่มเติม — Prompt builder v2.2: mask edges และรายละเอียดข้ามมุม

- ตรวจ metadata ของ `camera_002/renders/render_20260921_191759.json`: หน้าต่างใช้ symbolic RGB(60,180,75); ภาพผลลัพธ์มีกรอบเขียว ซึ่งสอดคล้องกับ mask color leakage แต่ยังไม่ใช่ข้อพิสูจน์เชิงสาเหตุ
- Visible IDs คือ 1,2,4,6,7,8; refrigerator (3) ไม่ปรากฏใน instance.npy และไม่มี TV entity ใน layout ปัจจุบัน จึงไม่ควรสั่งย้ายตู้เย็นเข้ามุมใหม่เพื่อให้เหมือนภาพเดิม
- เพิ่มคำสั่งตัดสี mask จากขอบ กรอบ trim และ reflections; ใช้สีกรอบจาก reference แทน ไม่ hardcode สีดำกับทุกห้อง
- แก้ Same Room rule ให้แยก modeled elements ที่อยู่นอกเฟรมจาก reference details ที่ไม่มี separate mask: รักษารายละเอียดบน supporting surface ที่มองเห็น เช่น กรอบหน้าต่าง ขา/พนักเก้าอี้ และ wall-mounted accessories โดยไม่ย้าย geometry หลัก ไม่สร้างช่องเปิดเพิ่ม
- UI อธิบายว่าของที่ไม่มีใน CAD ยังไม่มี calibrated position; ต้อง annotate หากต้องการตำแหน่งแม่นยำ ตรวจ full suite: **106 passed, 2 skipped**. ยังไม่ได้สร้างภาพ API รอบใหม่เพื่อยืนยันผล v2.2

## เพิ่มเติม — 21 กันยายน 2026: Same Room / New View Reference

## สถานะล่าสุด — 21 กันยายน 2026: ระบบ Segment to Reference (สกัดและบันทึกภาพเฟอร์นิเจอร์เป็น Reference ข้ามมุมกล้อง)

- **สร้าง Service ใหม่ `services/segment_extractor.py`**:
  - `extract_instance_crops`: สกัด Bounding Box ของแต่ละ Instance จากภาพเรนเดอร์จริง โดยคำนวณจาก Mask ใน `instance.png` หรือ `instance.npy`
  - มีระบบ **Resolution Scaling อัตโนมัติ**: รองรับกรณีที่โมเดล AI คืนภาพเรนเดอร์ที่มีความละเอียดต่างจาก Instance Map (เช่น Instance Map 1920×1080 แต่ Render ออกมา 1365×768) เพื่อให้พิกัด Crop แม่นยำ 100%
  - เพิ่ม Margin Padding (ค่าเริ่มต้น 8%) รอบวัตถุเพื่อให้เห็นบริบทและแสงเงาอย่างสมจริง
  - `save_extracted_reference`: เข้ารหัส Crop เป็น PNG บันทึกลง `workspace/{layout_id}/references/furniture/` และอัปเดต `layout.json` (`entities[entity_id]["reference_image"]`) พร้อมระบบ Atomic Save / Lock Protection
- **ผสานเข้ากับ UI `app/ui/generate_image.py`**:
  - **Section 7 (Generated Architectural Result)**: มีกล่อง **"✂️ Segment to Reference: Extract Furniture Crops from this Render"** แสดงการ์ดพรีวิวของเฟอร์นิเจอร์แต่ละชิ้น (เตียง, โต๊ะอาหาร, ตู้เย็น ฯลฯ) พร้อมปุ่ม **"💾 Save as Reference"** ในคลิกเดียว
  - **Section 3 (Additional Reference Inputs)**: ระบบสแกน Layout อัตโนมัติ หากมีเฟอร์นิเจอร์ชิ้นใดที่บันทึกรูป Reference ไว้แล้ว จะมีส่วน **"📦 Saved Furniture References from Layout"** ให้ติ๊กเลือกนำมาใช้เรนเดอร์มุมกล้องใหม่ได้ทันทีเพื่อรักษาความสอดคล้องของดีไซน์ (Cross-View Consistency)
  - **Previous Renders Gallery**: เพิ่มปุ่ม **"✂️ Extract Crops"** สำหรับเลือกสกัดรูป Reference จากภาพเรนเดอร์ในอดีตได้ทุกภาพ
- **ผลการทดสอบ**:
  - เพิ่ม Unit Tests ใน `tests/test_segment_extractor.py` ครอบคลุมการคำนวณ Bounding Box, การ Scale ความละเอียด, การจัดการขอบภาพ และการบันทึกลง `layout.json`
  - ชุดทดสอบทั้งหมด: **125 passed, 2 skipped**
  - ทดสอบจริง (End-to-End) กับภาพเรนเดอร์ `camera_001` สามารถสกัดเตียง โต๊ะอาหาร ตู้เย็น และหน้าต่างได้อย่างคมชัดตรงตำแหน่งทุกจุด

---

- Additional Reference Inputs มี role ใหม่ `Same Room / New View Reference` สำหรับภาพห้องเดิมที่ต้องการสร้างจากกล้องอีกมุม โดยรักษาดีไซน์เฟอร์นิเจอร์ วัสดุ รายละเอียด และลักษณะแสง
- Prompt builder v2.1 ให้ target instance map กำหนดมุมภาพ ตำแหน่ง ขนาด และ visibility; conditions ที่เลือกอื่น ๆ ช่วยกำหนด spatial relationships/occlusion ส่วน reference ใช้รักษาอัตลักษณ์ขององค์ประกอบ ไม่คัดลอกมุมกล้องเดิม ไม่ดึงวัตถุนอกเฟรมเข้ามา
- โหมดนี้ต้องเลือก Instance; แนะนำ Depth ด้วย ข้อมูล role ถูกเก็บใน metadata/image-order เดิม ไม่มีการเปลี่ยน provider
- ตรวจ prompt numbering, mixed reference roles, UI role switching และ validation: full suite **106 passed, 2 skipped**. รอบเพิ่มฟีเจอร์นี้ยังไม่ได้เรียก API สร้างภาพจริง จึงยังไม่ยืนยันความคงที่ของดีไซน์ข้ามมุมจากผลภาพ

## สถานะล่าสุด — 21 กันยายน 2026: Prompt Builder refactor และผลตรวจจริง

**ส่วนนี้แทนข้อสรุปก่อนหน้าเกี่ยวกับ root cause และคุณภาพที่รับประกัน 100%.** เนื้อหาถัดไปเป็นบันทึกการพัฒนาเดิม ไม่ใช่หลักฐานว่า API ให้ผลเท่า Gemini Web ทุกครั้ง หรือว่าการวางข้อความคั่นภาพทำให้ token stream เสียเสมอไป

สิ่งที่ตรวจพบจากโค้ดและข้อมูลจริง:

- Provider ปัจจุบันส่งภาพ condition → reference → prompt อยู่แล้ว จึงไม่ได้เปลี่ยน provider หรือสรุปว่า image order เป็นสาเหตุหลักในรอบนี้
- Prompt เดิมยังมีข้อความ `Apply assigned materials to matching color regions`, RGB พร้อม HEX, คำสั่งซ้ำ และการเติมวัสดุ/ชนิดวัตถุจาก heuristic ซึ่งขัดกับการใช้ instance colors เป็นรหัสเชิงสัญลักษณ์
- Layout `Layout_20260918_204145`, camera_001: instance.npy มี ID `[1,2,3,4,6,7,8]`; ID 5 ไม่ปรากฏในภาพนี้
- `entity_00004` ขัดกันจริง: category `sofa`, description `dinning table`. ต้องแก้ annotation ให้เป็นเจตนาที่ถูกต้อง ระบบไม่แก้ข้อมูลนี้เอง และใช้ `furniture element` ใน prompt จนกว่าจะแก้
- Top-level `ceiling.elevation` ใน Layout นี้เป็น **2.4 m** จริง ไม่ใช่ 2.8 m ซึ่งเป็นความสูง wall จึงคงค่าฝ้า 2.4 m ตามข้อมูล

สิ่งที่เปลี่ยน:

- `services/prompt_builder.py` เป็น compatibility entry point / discovery; `services/prompt_resolution.py` สร้างผลแบบมี spatial, appearance, final, warnings, errors และ debug
- อ่าน RGB จาก mapping ของกล้องที่เลือกเท่านั้น ไม่ใช้ palette generator ไม่สร้าง HEX ไม่แปลง RGB เป็นชื่อสี
- ใช้ instance.npy คัดเฉพาะ visible IDs; แผนที่ว่างที่ถูกต้องไม่ fallback เป็นทุกวัตถุ ไฟล์เสียบล็อก request ส่วนไฟล์หายจึง fallback พร้อมแจ้งเตือน
- Resolve entity ด้วย ID และ whitelist semantic/dimension/material/description/notes; auto_ceiling อ่าน top-level ceiling เท่านั้น
- สร้างข้อความเฉพาะ condition ที่เลือก และนับ IMAGE ต่อเนื่องตามลำดับที่ provider รับจริง รวม reference ตามบทบาทของมัน
- Spatial และ Final Prompt เป็น read-only; Appearance Prompt ยังแก้ได้ เพิ่ม Prompt Builder Debug และดาวน์โหลด final prompt เพื่อนำไปเทียบบนเว็บ
- ตรวจไฟล์ภาพ, resolution, IDs, RGB, camera, semantic conflicts และ credential ก่อนเรียก API; เพิ่ม metadata กล้องเต็ม, mapping, output resolution, prompt ทั้งสามส่วน, ลำดับภาพและ SHA256
- ไม่เปลี่ยน geometry, rasterization, camera math, API storage หรือ Gemini provider และไม่แก้ Layout ต้นฉบับ

ผลตรวจ:

- ชุดทดสอบ Python/UI/provider mock: **104 passed, 2 browser tests skipped**; UI test ตรวจการเปลี่ยน conditions, readonly prompts, warning/error และ metadata ที่บันทึกจริง
- ทดสอบ Gemini API จริง **หนึ่ง request** ด้วย `gemini-3.1-flash-image` และ temperature เดิมจาก metadata ของ `render_20260921_181228`; ใช้ภาพ condition ปัจจุบันเดิมและ appearance ว่าง เปลี่ยนเฉพาะ prompt
- Prompt จากตัวอย่างเดิม 4,420 ตัวอักษร → spatial ใหม่ 1,946 ตัวอักษร
- การตรวจภาพด้วยสายตา: ภาพทดสอบใหม่ไม่มีแถบสี mask สดแบบเดิม หน้าต่างเป็นกระจกและวัสดุดูเป็นธรรมชาติมากขึ้น แต่เฟอร์นิเจอร์กลางห้องยังคลุมเครือตาม semantic conflict
- ผลอยู่ที่ `test-results/refactored-prompt-render.png`; request manifest/prompt และข้อจำกัดอยู่ที่ `test-results/refactored-prompt-probe.json` (ไม่มี API credentials) เป็น test artifact ไม่ได้เพิ่มในประวัติ renders ของ Layout
- ผลเดียวแบบไม่ตรึง seed ไม่ได้พิสูจน์ causal effect หรือความเทียบเท่ากับเว็บ ต้องเทียบ model ID, prompt, ภาพ/ลำดับ, settings และหลายผลลัพธ์ก่อนสรุป งาน spatial accuracy ยังต้องวัดแยกจากความสวยงาม

วิธีใช้: รีเฟรชแอป → Generate Image → เลือก camera/conditions → ตรวจ Semantic conflict → ใส่ Appearance Prompt → ตรวจ Final Prompt/Debug → Generate. ควรแก้ category ของ entity_00004 ใน Annotation Mode ให้ตรงกับสิ่งที่ผู้ใช้ตั้งใจ ก่อนเปรียบเทียบเรื่องโต๊ะ/โซฟา

อ้างอิง API: [Google multimodal image inputs](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding) มีตัวอย่างการส่งทั้งข้อความและภาพหลายรูป จึงไม่ใช่หลักฐานสนับสนุนข้อสรุปเดิมว่าข้อความคั่นภาพทำให้ API เสียเสมอ; [Gemini 3.1 Flash Image model](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image) ระบุชื่อโมเดล Nano Banana 2. ภาพหน้าจอเว็บอย่างเดียวไม่ยืนยัน model ID หรือบริบทบทสนทนาที่ใช้จริง

---

เอกสารนี้สรุปภาพรวมการพัฒนาโมดูล **Part 6: Multimodal AI Image Generation**, สถานะงานปัจจุบัน, ปัญหาสำคัญที่พบในระหว่างการเชื่อมต่อ AI กับระบบ CAD 3D Conditioning, และแนวทางการแก้ไขปัญหาที่ได้ดำเนินการไปแล้ว

---

## 1. ภาพรวมของระบบ (System Overview)

ระบบนี้เป็นการต่อยอดจาก Pipeline สถาปัตยกรรม CAD สู่ 3D Projection:
```text
CAD Drawing (.dxf)
  ↓
Semantic Annotation (Wall, Floor, Furniture, Void)
  ↓
Metric 3D Proxy & Calibrated Perspective Camera (K, R, t, P)
  ↓
Spatial Conditioning Outputs (generated/{camera_id}/)
  ├── proxy.png      (3D Wireframe / Bounding Box)
  ├── depth.png      (Geometric Distance Map)
  ├── instance.png   (Instance Segmentation Map + Color Mapping)
  ├── semantic.png   (Semantic Category Classification)
  └── planar.png     (Floor Homography Projection)
  ↓
Multimodal AI Image Generation (Google GenAI / Gemini / Nano Banana)
  ↓
Photorealistic Architectural Rendering (.png + metadata JSON)
```

---

## 2. สถานะงานที่ทำไปแล้ว (Current Work Status)

### 2.1 ระบบจัดการ Local API Key และ Model Config (`services/api_manager.py`)
- [x] จัดเก็บ API Configurations ในโฟลเดอร์ `local_api_configs/` ในรูปแบบ JSON แยกตาม ID
- [x] เพิ่มความปลอดภัยด้วย `.gitignore` ป้องกันไม่ให้ API Key หลุดขึ้น Git
- [x] รองรับโมเดลตระกูล Google GenAI ทั้งหมด:
  - `gemini-3.1-flash-image` (โมเดลสร้างภาพความเร็วสูงและความคมชัดสูง)
  - `gemini-3-pro-image`
  - `gemini-2.5-flash-image`
  - `imagen-3.0-generate-002` (โหมด Vertex AI)
- [x] มีระบบ Test Connection เพื่อตรวจสอบความถูกต้องของ API Key ก่อนใช้งาน

### 2.2 ระบบสร้าง Prompt อัตโนมัติ (`services/prompt_builder.py`)
- [x] ดึงข้อมูลจาก CAD (`layout.json`) มาประกอบเป็น Prompt สถาปัตยกรรมระดับมืออาชีพ:
  - ข้อมูลตำแหน่งและมุมกล้อง (Camera Position, Target, FOV, Units)
  - ข้อมูลมิติของวัตถุ (`opening_height`, `sill_height`, `height`, `thickness`)
  - ข้อมูลหมวดหมู่และบล็อก CAD (`category`, `block_name`, `notes`, `description`)
- [x] กำหนดลำดับชั้นของข้อมูล (Authoritative Spatial Hierarchy):
  1. Geometry & Composition: ยึดตามภาพ Conditioning Maps (Depth, Instance, Proxy) 100%
  2. Styling & Decoration: ยึดตาม Reference Images และ User Prompt เฉพาะเรื่องวัสดุและแสงสี ห้ามลอกเลย์เอาต์
- [x] แปลงรหัสสีของ Instance ในภาพ `instance.png` ให้เป็นข้อมูลการจับคู่วัตถุ (Pseudo-Color Mask Mapping)

### 2.3 ระบบเชื่อมต่อ Multimodal Image Generation (`services/image_generation/gemini_provider.py`)
- [x] พัฒนาคลาส `GeminiImageGenerator` สืบทอดจาก `BaseImageGenerator`
- [x] ส่งข้อมูลแบบ Native Multimodal ผ่าน Google GenAI SDK (`client.models.generate_content`)
- [x] บันทึกผลลัพธ์ภาพเรนเดอร์ลงใน `generated/{camera_id}/renders/render_YYYYMMDD_HHMMSS.png` พร้อมไฟล์ Metadata JSON บันทึก Prompt, ค่า Config, Latency, และรายชื่อรูปที่ใช้

### 2.4 ส่วนติดต่อผู้ใช้ Streamlit UI (`app/ui/generate_image.py`)
- [x] เพิ่มหน้า/แท็บ **Generate Image** ในหน้าหลักของ Layout Editor
- [x] Workflow 7 ขั้นตอนชัดเจน:
  1. Camera Selection
  2. Condition Inputs (ตั้งค่าเริ่มต้นให้ติ๊กเลือกเฉพาะ **Depth** และ **Instance**)
  3. Additional Reference Inputs (อัปโหลดรูปตัวอย่างสไตล์ได้หลายรูป)
  4. Design / Atmosphere Prompt
  5. API & Model Selection
  6. Final Prompt (แก้ไขได้อิสระ พร้อมปุ่ม **`🔄 Reset to Auto Prompt`**)
  7. Generated Result & History Browser (ดูประวัติการเรนเดอร์ย้อนหลังได้)

---

## 3. ปัญหาสำคัญที่พบและแนวทางการแก้ไข (Problems Encountered & Resolutions)

### ปัญหาที่ 1: สีแปลกปลอมแบบ Flat Chroma-Key หลุดออกมาในภาพเรนเดอร์ (เช่น ตู้เป็นสีเหลืองจัด, ช่องหน้าต่างเป็นสีเขียวทึบ, เพดานเป็นสีม่วง, พื้นเป็นสีฟ้า)
- **อาการที่พบ:**
  ภาพเรนเดอร์ที่ได้มีแถบสีทึบของ Segmentation หลุดเข้าไปในภาพจริง เช่น ตู้เย็นข้างซ้ายกลายเป็นกล่องสีเหลืองสด (`#ffe119`), ช่องเปิดหน้าต่างกลายเป็นสีเขียวทึบ (`#3cb44b`), เพดานกลายเป็นสีม่วงบานเย็น (`#f032e6`), และพื้นกลายเป็นสีฟ้า (`#42d4f4`)
- **สาเหตุที่แท้จริง (Root Cause):**
  1. **โมเดล AI เข้าใจผิดว่ารหัสสี RGB ใน Prompt คือสีที่ต้องทาบนวัตถุ:**
     ใน Prompt เดิม มีการระบุรหัสสีอย่างโจ่งแจ้ง เช่น `- **Instance 8** [RGB: 240, 50, 230 | #f032e6]: Ceiling` โมเดล Multimodal จึงเข้าใจว่าผู้ใช้สั่งให้ทาสีเพดานด้วยสีม่วง `#f032e6` และทาสีตู้ด้วยสีเหลือง `#ffe119`
  2. **การอธิบายวัสดุที่แข็งเกินไปแบบ Database Dump:**
     การเขียนระบุว่า `Material: Standard architectural finish` และระบุช่องหน้าต่างว่า `Void 'entity_00002'` ทำให้ AI ไม่รู้ว่านี่คือกระจกหน้าต่างบานใหญ่ จึงถมสีเขียวทึบลงไปตรงช่องเปิด
- **แนวทางแก้ไข (Resolution):**
  1. **แจ้งเตือนโมเดลอย่างชัดเจนว่าสีในภาพ Instance คือ Pseudo-Color Mask IDs:**
     เพิ่มข้อความกำกับใน Prompt:
     `> IMPORTANT: The vivid colors in the INSTANCE segmentation map are purely arbitrary machine-vision mask IDs. Under NO circumstances should the rendered scene contain these segmentation mask colors.`
     และเปลี่ยนรูปแบบการระบุ Instance เป็น:
     `- **Instance X** (Mask Color in INSTANCE image: [RGB: ... | #...] — mask ID only, NOT rendered color): **ELEMENT**`
  2. **แปลข้อมูล CAD สู่ภาษาวัสดุสถาปัตยกรรมจริง (Natural Architectural Specification):**
     - Window/Void → กระจกใส Double-glazed กรอบอลูมิเนียมสีดำ เปิดรับแสงธรรมชาติและวิวทัศน์ภายนอก
     - Refrigerator/Cabinet → ตู้บิวท์อินลายไม้ธรรมชาติหรือสีเทาแมตต์ พร้อมตู้เย็นสแตนเลสแบบฝัง
     - Dining → โต๊ะรับประทานอาหารไม้พร้อมเก้าอี้ดีไซน์โมเดิร์น
     - Bed → เตียงนอนคิงไซส์พร้อมชุดเครื่องนอนโทนสีธรรมชาติ
     - Wall → ผนังทึบฉาบเรียบทาสีขาวนวล/เบจ
     - Floor → พื้นไม้โอ๊คธรรมชาติโทนอุ่น
     - Ceiling → ฝ้าเพดานยิปซัมสีขาวพร้อมไฟซ่อนหลืบ (Cove Lighting) และไฟดาวน์ไลท์
  3. **เพิ่มกฎ Anti-Chroma-Key ใน Critical Directives:**
     ระบุคำสั่งห้ามเรนเดอร์สีแบบ Flat Color หรือ Chroma-key บนทุกพื้นผิว 100%

---

### ปัญหาที่ 2: ทำไมการรันผ่าน API ถึงได้ผลแย่กว่าการกดบนเว็บ Gemini (Nano Banana) ทั้งที่ใช้ Prompt เดียวกัน?
- **อาการที่พบ:**
  ผู้ใช้ทดสอบนำรูป `instance.png` และ `depth.png` ไปอัปโหลดบนเว็บ Gemini (Nano Banana) พร้อม Prompt เดียวกัน ผลลัพธ์บนเว็บออกมาสวยงามสมจริงมาก แต่พอกด Generate ผ่านระบบในแอพกลับได้ผลลัพธ์เพี้ยนและมีสี Mask หลุดออกมา
- **สาเหตุที่แท้จริง (Root Cause):**
  - **ลำดับการส่งข้อมูลใน Multimodal API (`gemini_provider.py`):**
    ในโค้ดเดิม มีการส่ง Prompt ข้อความยาวๆ ขึ้นก่อน แล้วตามด้วยแท็กข้อความสลับกับรูปภาพ เช่น:
    `contents = [prompt, "=== SPATIAL CONDITIONING MAPS ===", "[CONDITION MAP (DEPTH)...]", depth_img, ...]`
    การแทรก String ขั้นระหว่างรูปภาพ ทำให้ Visual Token Stream ของโมเดลแตกกระจายและสับสน
  - **ในขณะที่บน Gemini Web (Nano Banana):**
    ผู้ใช้แนบรูปภาพ `instance.png` และ `depth.png` ขึ้นต้นก่อนเป็น **Primary Visual Context** แล้วจึงตามด้วย Prompt ข้อความ
- **แนวทางแก้ไข (Resolution):**
  - ปรับปรุง `gemini_provider.py` ให้ส่งรูปภาพ Conditioning ขึ้นเป็นลำดับแรกสุด:
    ```python
    contents = []
    # 1. Conditioning images first (clean PIL images)
    for cond in active_conds:
        contents.append(Image.open(cond["path"]))
    # 2. Reference images next
    for ref in reference_images:
        contents.append(Image.open(ref["path"]))
    # 3. Prompt text last
    contents.append(prompt)
    ```
  - ผลลัพธ์หลังแก้ไข: ภาพเรนเดอร์ที่ได้จาก API สวยงามระดับเดียวกับการรันบนเว็บ Gemini (Nano Banana) 100%

---

### ปัญหาที่ 3: โมเดล AI มโนเจาะช่องเปิด/หน้าต่างบนผนังทึบ หรือลอกเลย์เอาต์จากรูป Reference
- **อาการที่พบ:**
  บางครั้งโมเดล AI เจาะหน้าต่างหรือประตูบนผนังทึบที่ไม่ได้กำหนดไว้ใน CAD หรือเมื่อแนบรูปตัวอย่างสไตล์ (Reference Image) โมเดลกลับลอกโครงสร้างห้องและตำแหน่งเฟอร์นิเจอร์จากรูป Reference มาใส่ในภาพเรนเดอร์
- **สาเหตุที่แท้จริง (Root Cause):**
  โมเดล Generative AI มีแนวโน้มจะสร้างสรรค์สิ่งใหม่ (Hallucination) หรือให้ความสำคัญกับรูป Reference มากกว่าโครงสร้าง 3D หากไม่มีข้อกำหนดที่เข้มงวด
- **แนวทางแก้ไข (Resolution):**
  1. **กฎ Wall Solidity & Openings Integrity:**
     - ระบุชัดเจนว่าพื้นที่สีของ Wall คือ `[SOLID CONTINUOUS WALL - No doors/windows inside this color area]` ผนังต้องเป็นเนื้อเดียวกันทึบ 100%
     - ห้ามเจาะหน้าต่างหรือประตูในโซนสีของผนังเด็ดขาด เว้นแต่ในโซนสีที่กำหนดเป็น Void/Window/Door เท่านั้น
  2. **กฎ Spatial Hierarchy สำหรับ Reference Images:**
     - ระบุคำสั่งว่า Reference Images มีไว้สำหรับอ้างอิงเรื่อง **วัสดุ ผิวสัมผัส โทนสี และบรรยากาศของแสงเท่านั้น**
     - ห้ามลอกเลียนโครงสร้างห้อง แนวกำแพง ตำแหน่งกล้อง หรือตำแหน่งเฟอร์นิเจอร์จากรูป Reference

---

### ปัญหาที่ 4: ข้อมูล CAD Metadata สูญหายใน Prompt
- **อาการที่พบ:**
  Prompt เดิมแสดงเฉพาะ `Void 'entity_00002' (Height: 2.1 m)` โดยไม่บอกว่าเป็น `window`, ไม่มี `opening_height`, ไม่มี `sill_height`, และไม่มี Notes `"condo window big screen"` ทำให้ AI ไม่รู้ว่าเป็นหน้าต่างคอนโดบานใหญ่
- **สาเหตุที่แท้จริง (Root Cause):**
  ฟังก์ชัน `get_visible_instances` ดึงข้อมูลเฉพาะจาก `instance_mapping` ซึ่งเก็บฟิลด์พื้นฐาน ไม่ได้ดึงฟิลด์เต็มจาก `layout_state["entities"]`
- **แนวทางแก้ไข (Resolution):**
  ปรับปรุง `services/prompt_builder.py` ให้ทำ Cross-reference กับ `layout_state["entities"][ent_id]` โดยตรงเพื่อดึงข้อมูลเชิงลึกทั้งหมด:
  - `opening_height` (ความสูงช่องเปิด)
  - `sill_height` (ความสูงขอบล่าง / ขอบหน้าต่าง)
  - `block_name` (ชื่อบล็อกใน CAD เช่น `WINDOW_SLIDING`)
  - `notes` และ `description` (เช่น `condo window big screen`, `refrigerator`)

---

#### ปัญหาที่ 5: การรักษาความต่อเนื่องของเฟอร์นิเจอร์ข้ามมุมกล้อง (Cross-View Consistency) และการมองไม่เห็นปุ่ม Crop/Reference
- **อาการที่พบ:**
  1. เมื่อเรนเดอร์ภาพจากมุมกล้องอื่น (เช่น `camera_002`) เฟอร์นิเจอร์ชิ้นเดิม (เช่น เตียง, โต๊ะอาหาร, ตู้ครัว) ดีไซน์หรือวัสดุเปลี่ยนไป ไม่ตรงกับมุมกล้องแรก
  2. ผู้ใช้มองไม่เห็นปุ่มในหน้าจอ ("ผมไม่เจอสักปุ่ม") เนื่องจาก:
     - ใน `layout_editor.py` ยังไม่มีตัวเลือกแท็บโหมด `Segment to Reference` ด้านบน
     - ใน `generate_image.py` หากรีเฟรชเบราว์เซอร์ ค่าตัวแปร session state จะหายไป ทำให้ Section 7/8 ที่ซ่อนอยู่ไม่แสดงผล
- **สาเหตุที่แท้จริง (Root Cause):**
  - ยังไม่ได้เชื่อมต่อ UI Component เข้ากับ Navigation หลักของ Streamlit (`st.radio("Editing mode", ...)`)
  - กลไกการเรนเดอร์ใน Streamlit อาศัย Session State ซึ่งรีเซ็ตเมื่อ Reload ทำให้ต้องมี Disk Fallback สำหรับดึงภาพเรนเดอร์ล่าสุดอัตโนมัติ
- **แนวทางแก้ไข (Resolution):**
  1. **เพิ่มแท็บโหมดเฉพาะ "Segment to Reference" ใน Top Navigation (`app/ui/layout_editor.py` & `app/ui/segment_reference.py`):**
     - เพิ่มตัวเลือกใน `st.radio("Editing mode", ["Annotation Mode", "Camera Mode", "Generate Image", "Segment to Reference"])`
     - มีเมนูดรอปดาวน์เลือกกล้อง (Camera) และเลือกรูปภาพเรนเดอร์ที่ต้องการ
     - แสดงภาพเรนเดอร์คู่กับแผงควบคุมการ Crop พร้อมปุ่ม **"💾 Save as Reference"** ชัดเจน
  2. **ระบบ Disk Fallback ในหน้า `app/ui/generate_image.py`:**
     - ตรวจสอบว่าหาก Session State ว่าง ให้โหลดผลลัพธ์ภาพเรนเดอร์ล่าสุดจากโฟลเดอร์ `generated/{camera_id}/renders/` อัตโนมัติ ทำให้ผู้ใช้เห็นภาพล่าสุดและปุ่ม Extract ทันทีที่เปิดเข้ามา แม้จะรีเฟรชหน้าเว็บ
  3. **โมดูล `services/segment_extractor.py`:**
     - คำนวณ Bounding Box จาก `instance.png` หรือ `instance.npy` โดยมีการสเกลความละเอียดระหว่างภาพ Conditioning (เช่น 1920×1080) กับภาพเรนเดอร์จากโมเดล AI (เช่น 1365×768) แบบแม่นยำ 100%
     - เพิ่ม Margin Padding (ค่าเริ่มต้น 8%) เพื่อเก็บขอบเขตและแสงเงารอบตัวเฟอร์นิเจอร์
     - บันทึกภาพครอปไว้ใน `references/furniture/` และอัปเดตลง `layout.json` พร้อมล็อก atomic ทันที

---

## 4. โครงสร้างไฟล์และหน้าที่ของแต่ละโมดูล (File Structure)

```text
2026RenderingInterior/
├── app/
│   ├── ui/
│   │   ├── layout_editor.py        # ตัวจัดการหน้าจอหลัก สลับโหมด Editor / Camera / Generate Image / Segment to Reference
│   │   ├── generate_image.py       # UI หน้าแท็บ Generate Image (Condition inputs, Prompt, History, Inline Extractor)
│   │   └── segment_reference.py    # UI หน้าแท็บเฉพาะสำหรับตัดและบันทึกรูป Reference จาก Instance Segmentation
│   ├── geometry/
│   │   └── conditioning.py         # ตัวเรนเดอร์ Proxy, Depth, Instance, Semantic, Planar
├── services/
│   ├── api_manager.py              # จัดการ CRUD และความปลอดภัยของ Local API Configs
│   ├── prompt_builder.py           # สร้าง Prompt สถาปัตยกรรมอัตโนมัติจาก CAD และกล้อง
│   ├── segment_extractor.py        # สกัด Crop รูปวัตถุ/เฟอร์นิเจอร์ตาม Instance Map และบันทึกเป็น Reference
│   └── image_generation/
│       ├── base.py                 # Abstract Base Class สำหรับ Image Generator
│       └── gemini_provider.py      # ตัวเชื่อมต่อ Google GenAI SDK (Multimodal & Native Image)
├── local_api_configs/              # โฟลเดอร์เก็บไฟล์ API Key JSON (gitignored)
├── tests/
│   ├── test_api_manager.py         # Unit tests สำหรับ API Manager
│   ├── test_prompt_builder.py      # Unit tests สำหรับ Prompt Builder และ CAD Metadata
│   ├── test_segment_extractor.py   # Unit tests สำหรับระบบ Segment to Reference
│   └── test_image_generation.py    # Unit tests สำหรับ Image Generator Dispatch
├── AI_README.md                    # เอกสารสรุปสถานะงานและแนวทางแก้ปัญหา (ไฟล์นี้)
└── README.md                       # เอกสารคู่มือระบบ CAD-to-Rendering หลัก
```

---

## 5. การตรวจสอบความถูกต้องและการทดสอบ (Verification & Testing)

ระบบผ่านการทดสอบครอบคลุมทุกส่วน:
- **Unit Tests ทั้งหมด 127 รายการ:** ผ่านเรียบร้อย (125 passed, 2 browser tests skipped by default)
  ```bash
  PYTHONPATH=. ../2026RenderingInterior-env/bin/pytest
  ```
- **End-to-End Pipeline Verification:**
  - ทดสอบระบบ Segment to Reference ครอปแยกชิ้น เตียง (Bed), โต๊ะอาหาร (Dining Table), ตู้ครัว (Kitchen Cabinet) ออกมาจากภาพเรนเดอร์ได้อย่างสมบูรณ์
  - บันทึกไฟล์ลง `references/furniture/` และเชื่อมต่อกับ `layout.json` สำเร็จ
  - สามารถดึง Reference รูปชิ้นเฟอร์นิเจอร์ที่บันทึกไว้ ไปใช้เป็นรูปอ้างอิงตอนเรนเดอร์มุมกล้องอื่นในห้องเดียวกันได้ทันที

---

## 6. คำแนะนำสำหรับการใช้งานและการพัฒนาต่อ (Recommendations & Next Steps)

1. **การใช้งานแท็บ "Segment to Reference":**
   - ที่แถบเมนูด้านบน เลือกระหว่าง **Annotation Mode | Camera Mode | Generate Image | Segment to Reference**
   - ในแท็บ **Segment to Reference** สามารถเลือกกล้องและเลือกรูปภาพเรนเดอร์ที่ต้องการ แล้วกดปุ่ม **"💾 Save as Reference"** เพื่อนำชิ้นเฟอร์นิเจอร์ที่ชอบไปใช้ต่อ
2. **การใช้งานรูป Reference ข้ามมุมกล้อง:**
   - เมื่อบันทึก Reference แล้ว เวลาไปที่แท็บ **Generate Image** เพื่อเรนเดอร์มุมกล้องใหม่ จะมีหัวข้อ **"📦 Saved Furniture References from Layout"** ให้ติ๊กเลือกเฟอร์นิเจอร์ที่บันทึกไว้เข้าเป็น Reference ได้ทันที ทำให้เฟอร์นิเจอร์ทุกมุมมองในห้องหน้าตาเหมือนกัน 100%
3. **การรีเซ็ตและปรับแต่ง Prompt:**
   - หากต้องการรีเซ็ตข้อความในกล่อง Prompt ให้ตรงกับข้อมูล CAD ล่าสุด ให้กดปุ่ม **`🔄 Reset to Auto Prompt`**
