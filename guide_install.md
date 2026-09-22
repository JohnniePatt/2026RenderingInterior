# คู่มือการติดตั้งระบบ CAD-to-Rendering Research Application บนเครื่องใหม่ (Installation Guide)

คู่มือนี้จัดทำขึ้นเพื่อให้สามารถนำโค้ดไปติดตั้งและรันบนเครื่องคอมพิวเตอร์เครื่องอื่นได้อย่างราบรื่น ครอบคลุมตั้งแต่การตั้งค่าระบบปฏิบัติการ, การจัดการ Virtual Environment, การแก้ปัญหาเน็ตเวิร์กของ WSL2, การดาวน์โหลดโมเดล AI (SAM 3) ไปจนถึงการเปิดใช้งานเว็บแอปพลิเคชัน

---

## 1. ข้อกำหนดเบื้องต้นของระบบ (System Requirements & Prerequisites)

1. **ระบบปฏิบัติการ:**
   - Windows 10 หรือ Windows 11 (รองรับ WSL2)
   - ติดตั้ง **WSL2** พร้อม **Ubuntu 22.04 LTS หรือ 24.04 LTS**
   - Python 3.11 หรือ 3.12 (แนะนำ 3.12)
2. **บัญชีและการเข้าถึง (Accounts & API Keys):**
   - **Hugging Face Account:**
     - เข้าไปขอสิทธิ์เข้าใช้งานโมเดล Gated ที่: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) (รอการอนุมัติ)
     - สร้าง User Access Token (Read Permission) ที่: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - **Google Gemini API Key:**
     - สร้าง API Key จาก [Google AI Studio](https://aistudio.google.com/)
     - *สำคัญ:* สำหรับโมเดลสร้างภาพ (`gemini-3.1-flash-image` หรือ `gemini-2.5-flash-image`) จำเป็นต้องผูกบัตร/บัญชี Pay-as-you-go ใน Google AI Studio / Google Cloud Console เพราะ Free-tier มีโควตาภาพเป็น 0

---

## 2. การตั้งค่าเน็ตเวิร์กบน WSL2 (สำคัญมาก: ป้องกันปัญหาเน็ตหลุด / Errno 104)

หากเครื่องของคุณเปิดใช้งาน **Cloudflare WARP**, VPN หรือเครือข่ายที่มี MTU ต่ำ (1280) WSL2 มักจะเกิดปัญหา **TLS Handshake Hang / `[Errno 104] Connection reset by peer`** เมื่อเชื่อมต่อกับ Google API หรือ Hugging Face ให้แก้ไขดังนี้:

### 2.1 ปรับ MTU ของ WSL2 ให้ตรงกับ Host Network (1280)
เปิด Terminal ใน WSL (Ubuntu) แล้วรันคำสั่ง:
```bash
sudo ip link set dev eth0 mtu 1280
```

### 2.2 ตั้งค่าให้ปรับ MTU อัตโนมัติทุกครั้งที่เปิดเครื่อง (Persistent Service)
```bash
sudo bash -c 'cat << "EOF" > /etc/systemd/system/set-mtu.service
[Unit]
Description=Set MTU for eth0 to match host VPN/WARP
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/ip link set dev eth0 mtu 1280
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable set-mtu.service'
```

### 2.3 ตรวจสอบให้แน่ใจว่า IPv6 เปิดทำงานปกติ
เพื่อให้ Windows Browser สามารถเข้าใช้งานผ่าน `http://localhost:8501` ได้:
```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=0
```

---

## 3. การ Clone โค้ดและสร้าง Virtual Environment

เปิด WSL Terminal แล้วไปยังโฟลเดอร์ที่ต้องการเก็บโปรเจกต์:

```bash
# 1. ไปยังไดเรกทอรีทำงาน
cd ~/programming

# 2. Clone repository
git clone <YOUR_GIT_REPO_URL> 2026RenderingInterior
cd 2026RenderingInterior

# 3. สร้าง Python Virtual Environment
python3 -m venv 2026RenderingInterior-env

# 4. สลับเข้าใช้งาน Virtual Environment
source ../2026RenderingInterior-env/bin/activate
```

---

## 4. การติดตั้ง Dependencies (Python Packages)

รันคำสั่งติดตั้งแพ็กเกจตามลำดับดังนี้:

```bash
# อัปเกรด pip
pip install --upgrade pip

# 1. ติดตั้งแพ็กเกจหลักของระบบ (Core Dependencies)
pip install -r requirements.txt

# 2. ติดตั้ง Google GenAI SDK สำหรับ Gemini Image Generation
pip install google-genai==2.24.0

# 3. ติดตั้ง PyTorch (สำหรับ CPU หรือ CUDA ตามการ์ดจอของเครื่อง)
# กรณีใช้งาน CPU (มาตรฐาน):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# กรณีใช้งาน NVIDIA GPU (CUDA 12.4):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. ติดตั้ง Transformers และ SAM 3 Dependencies
pip install -r requirements-sam3.txt
pip install huggingface_hub
```

---

## 5. การดาวน์โหลดและติดตั้งโมเดล SAM 3 (`facebook/sam3`)

โมเดล SAM 3 มีขนาดประมาณ 3.28 GB ต้องดาวน์โหลดมาเก็บไว้ที่ `~/.cache/cad-rendering/sam3` ก่อนใช้งาน

### 5.1 บันทึก Hugging Face Token ใน WSL
แทนที่ `hf_xxxx...` ด้วย Access Token ของคุณ:
```bash
mkdir -p ~/.cache/huggingface
echo "hf_xxxx_YOUR_TOKEN_HERE" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

### 5.2 รันสคริปต์ดาวน์โหลดโมเดล
```bash
python scripts/setup_sam3.py
```
*หากดาวน์โหลดสำเร็จ จะมีข้อความแสดงว่า:*
`SAM 3 cached locally: /home/<username>/.cache/cad-rendering/sam3`

> [!TIP]
> **กรณีที่ดาวน์โหลดบน WSL ช้าหรือไม่เสถียร:**
> สามารถเปิด **PowerShell บน Windows** แล้วใช้คำสั่งนี้ดาวน์โหลดจาก Windows ส่งตรงเข้าโฟลเดอร์ของ WSL ได้ทันที:
> ```powershell
> pip install huggingface_hub
> python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3', local_dir=r'\\wsl.localhost\Ubuntu\home\<username>\.cache\cad-rendering\sam3', allow_patterns=['*.json','*.safetensors','*.txt','*.model'], token='hf_xxxx_YOUR_TOKEN_HERE')"
> ```

---

## 6. การทดสอบความถูกต้องของระบบ (Verification)

ก่อนเริ่มใช้งาน ให้รัน Unit Tests เพื่อยืนยันว่าการติดตั้งครบถ้วน:

```bash
python -m pytest tests/test_sam3_provider.py tests/test_automatic_correspondence.py tests/test_spatial_correspondence.py
```
ผลลัพธ์ควรขึ้น **`passed`** ครบทุกตัว

---

## 7. การเปิดใช้งานแอปพลิเคชัน (Running the Application)

รันคำสั่งเปิด Streamlit Server:

```bash
cd ~/programming/2026RenderingInterior/2026RenderingInterior
source ../2026RenderingInterior-env/bin/activate

streamlit run app.py --server.port 8501 --server.headless true
```

เปิดเบราว์เซอร์บน Windows เข้าไปที่:
👉 **`http://localhost:8501`**
*(หรือหากติดปัญหา Network ให้ใช้ `http://<WSL_IP>:8501` เช่น `http://172.21.36.77:8501`)*

---

## 8. ข้อแนะนำสำคัญสำหรับการใช้งานจริง (Best Practices)

### 8.1 การตั้งค่าความละเอียดกล้อง (Camera Resolution) ให้ตรงกับ Gemini API
ระบบ Gemini Image Generation มีขนาดความละเอียดภาพตายตัวตามสัดส่วน ดังนี้:
- **สัดส่วน 1:1** ➔ **1024 × 1024 พิกเซล**
- **สัดส่วน 16:9** ➔ **1376 × 768 พิกเซล**

> [!IMPORTANT]
> **การป้องกันปัญหา Resolution Mismatch ในแท็บ Spatial Validation (Part 7A):**
> - หากคุณต้องการใช้กล้องสัดส่วน **1:1** (เช่น `camera_005`): ให้ตั้งค่าความละเอียดของกล้องใน CAD เป็น **1024 × 1024**
> - หากคุณต้องการใช้กล้องสัดส่วน **16:9** (เช่น `camera_001`): ให้ตั้งค่าความละเอียดของกล้องใน CAD เป็น **1376 × 768**
> - เมื่อตั้งค่าแล้ว ให้กด **"Export Conditioning (PNG & NPY)"** ใน Camera Mode ก่อนทำการเรนเดอร์ ภาพ CAD กับภาพที่ได้จาก AI จะตรงกัน 100% และปุ่ม **Run Automatic Correspondence** จะทำงานได้ทันที

### 8.2 การดูผลลัพธ์ในแท็บ Spatial Validation
- หากกด `Run Automatic Correspondence` แล้วไม่เห็นภาพ Mask ด้านล่าง ให้ตรวจดูที่ช่องตัวกรอง:
  - **เอาเครื่องหมายถูกออกจากช่อง `[ ] Show cases needing attention only`**
  - รายการวัตถุที่ระบบตรวจจับได้สำเร็จทั้งหมดจะปรากฏขึ้นมาให้เลือกตรวจสอบ Overlay ได้ทันที

---

## 9. สรุปคำสั่งแบบ Quick Start (สำหรับเครื่องใหม่)

```bash
# 1. Setup WSL MTU
sudo ip link set dev eth0 mtu 1280

# 2. Setup Venv & Install
cd ~/programming/2026RenderingInterior
python3 -m venv ../2026RenderingInterior-env
source ../2026RenderingInterior-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install google-genai==2.24.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sam3.txt huggingface_hub

# 3. Setup HF Token & Download Model
mkdir -p ~/.cache/huggingface
echo "YOUR_HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
python scripts/setup_sam3.py

# 4. Run App
streamlit run app.py --server.port 8501 --server.headless true
```
