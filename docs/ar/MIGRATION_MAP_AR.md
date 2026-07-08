# خريطة تحويل MATLAB إلى Python

| ملف MATLAB | المشكلة/الدور | المقابل في Python |
|---|---|---|
| `MAIN.M` | تحميل البيانات، إنشاء النوافذ، التقسيم، التدريب، المقاييس، الرسومات | `run_reproduction.py` و`src/nfpsosc/experiment.py` |
| `CreateTimeSeriesData.m` | إنشاء المدخلات المتأخرة والهدف | `src/nfpsosc/data.py:create_lagged_data` |
| `CreateInitialFIS.m` | إنشاء FIS أولي، لكنه يستخدم FCM وليس SC | `src/nfpsosc/clustering.py:initialize_tsk_from_sc` |
| `GetFISParams.m` | تسطيح معاملات FIS | `src/nfpsosc/fis.py:TSKFIS.to_vector` |
| `SetFISParams.m` | إعادة المعاملات إلى FIS | `src/nfpsosc/fis.py:TSKFIS.from_vector` |
| `TrainFISCost.m` | تقييم العامل المضاعف باستخدام `std(error)` | مسار `legacy_intent` داخل `experiment.py` |
| `TrainAnfisUsingPSO.m` | PSO بعوامل مضاعفة وحدود وسرعة | `src/nfpsosc/optimizers.py` |
| `PlotResults.m` | الحقيقي/المتنبأ والبواقي | `src/nfpsosc/plots.py` |
| `RouletteWheelSelection.m` | خاص بالخوارزمية الجينية | غير مطلوب لمسار NF-PSO-SC الحالي |
| `TrainAnfisUsingGA.m` | بديل GA وليس جوهر الورقة الجديدة | لم يُحوّل في هذه المرحلة؛ سيضاف فقط إذا تقرر استخدامه baseline مستقلًا |

## أدوات إضافية للورقة الجديدة

- `run_radius_sweep.py`: علاقة نصف قطر SC بعدد القواعد والخطأ والحساسية.
- `run_noise_sensitivity.py`: مقاومة اضطراب المدخلات وربطها بـLipschitz/Jacobian.
- `run_seed_study.py`: التباين بين البذور العشوائية واستقرار PSO.
- `outputs/*/pso_history.csv`: التكلفة، السرعة، قطر السرب، انحراف المعاملات واصطدامات الحدود.
