# إعادة تنفيذ NF-PSO-SC بلغة Python

## الهدف

هذا المشروع لا ينسخ أكواد MATLAB سطرًا بسطر. بل يقدم إعادة تنفيذ شفافة وقابلة للاختبار للنموذج المقصود في رسالة الماستر، مع جمع المعلومات التي تحتاجها الورقة الجديدة:

- عدد القواعد ومعاملاتها.
- مقاييس التدريب والاختبار.
- منحنى تقارب PSO.
- سرعة الجسيمات وقطر السرب.
- تغير أفضل المعاملات واصطدامات الحدود.
- مجموع درجات تفعيل القواعد.
- معيار Jacobian والحساسية التجريبية للمدخلات.
- ملفات CSV للنتائج والتنبؤات والمعاملات.

## حقيقة مهمة عن ملفات MATLAB

اقرأ `MATLAB_CODE_AUDIT_AR.md` أولًا. الأكواد المرفقة غير متسقة مع بعضها ولا تشغّل NF-PSO-SC كما هو موصوف دون إصلاحات.

## بنية البيانات

الملف:

`data/yemen_tax_revenues_2002_2014.csv`

يحتوي 156 مشاهدة شهرية من يناير 2002 إلى ديسمبر 2014.

يستخدم المشروع خمس قيم سابقة للتنبؤ بالقيمة السادسة، بترتيب زمني يطابق جدول الملحق:

`[t-5, t-4, t-3, t-2, t-1] -> t`

بعد ذلك ينتج 151 نموذجًا إشرافيًا: 128 تدريبًا و23 اختبارًا عند تقسيم 85%/15%.

## التثبيت

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## تشغيل سريع للتحقق

```bash
python run_reproduction.py --protocol legacy_intent --quick
python run_reproduction.py --protocol research_baseline --quick
```

## تشغيل الإعداد الكامل

```bash
python run_reproduction.py --protocol legacy_intent --iterations 1000 --particles 25 --seed 123
python run_reproduction.py --protocol research_baseline --iterations 1000 --particles 25 --seed 123
```

## الفرق بين المسارين

### legacy_intent

- التطبيع باستخدام كامل السلسلة، كما في جداول الرسالة.
- SC بنصف قطر 0.55.
- PSO يبحث عن عوامل مضاعفة للمعاملات الأولية داخل `[-25,25]`.
- دالة الهدف الافتراضية هي الانحراف المعياري للبواقي، كما في `TrainFISCost.m`.
- يُستخدم فقط لتدقيق العمل القديم، لا كبروتوكول نهائي للورقة.

### research_baseline

- التطبيع من بيانات التدريب فقط.
- تقسيم زمني بلا خلط.
- Projected Constricted PSO.
- حدود منفصلة للمراكز، عروض العضوية، ومعاملات النتائج.
- دالة هدف تجمع RMSE وعقوبة حساسية Jacobian.
- مناسب كنقطة بداية للنموذج النظري الجديد، وليس نتيجة نهائية قبل التجارب الموسعة.

## المخرجات

داخل `outputs/<protocol>/`:

- `summary.json`: المقاييس والتشخيصات وإشارات التدقيق.
- `test_predictions.csv`: الحقيقي والمتنبأ والبواقي.
- `pso_history.csv`: تاريخ تقارب PSO وتشخيصاته.
- `parameters.csv`: المعاملات الأولية والنهائية.
- `test_actual_vs_predicted.png`.
- `test_residuals.png`.
- رسومات PSO للتكلفة والسرعة وقطر السرب والانجراف والحدود.

## الاختبارات

```bash
python -m pytest tests -q
```

`pytest` ليس ضمن المتطلبات التشغيلية الأساسية؛ ثبته فقط لتشغيل الاختبارات:

```bash
pip install pytest
```

## حدود المطابقة

- تنفيذ Subtractive Clustering هنا شفاف ومبني على إجراء Chiu القياسي.
- لا ندعي التطابق العددي الحرفي مع `genfis2` في إصدار MATLAB غير محدد.
- الهدف هو إعادة إنتاج علمية قابلة للتدقيق، لا استنساخ صندوق MATLAB الأسود.

## دراسات مخصصة للورقة الجديدة

### مسح نصف قطر SC

```bash
python run_radius_sweep.py
```

ينتج عدد القواعد، عدد المعاملات، RMSE، MAPE، Jacobian، الحساسية، فصل المراكز، والزمن.

### اختبار الضوضاء والحساسية

بعد تشغيل تجربة وحفظ `model.npz`:

```bash
python run_noise_sensitivity.py --experiment outputs/legacy_intent --trials 30
```

### دراسة البذور

```bash
python run_seed_study.py --protocol legacy_intent --iterations 200
```

## ملفات يجب قراءتها قبل استخدام النتائج

- `MATLAB_CODE_AUDIT_AR.md`
- `INITIAL_FINDINGS_AR.md`
- `MIGRATION_MAP_AR.md`
