# RDKit 分子描述符 (Descriptors) 分類說明

總共 **217 個描述符**，可分為以下幾大類：

## 1. 基本分子性質 (Basic Properties)
- **MolWt**: 分子量
- **ExactMolWt**: 精確分子量
- **HeavyAtomCount**: 重原子數
- **HeavyAtomMolWt**: 重原子分子量
- **NumHeteroatoms**: 雜原子數
- **NumValenceElectrons**: 價電子數
- **NumRadicalElectrons**: 自由基電子數
- **RingCount**: 環數
- **FractionCSP3**: Csp3 碳原子比例

## 2. 拓撲描述符 (Topological Descriptors)
- **Chi0, Chi1, Chi2n, Chi2v, Chi3n, Chi3v, Chi4n, Chi4v**: 分子連接性指數
- **Chi0n, Chi0v, Chi1n, Chi1v**: 標準化連接性指數
- **Kappa1, Kappa2, Kappa3**: Kappa 形狀指數
- **Phi**: 靈活性指數
- **BalabanJ**: Balaban J 指數
- **BertzCT**: Bertz 複雜度指數
- **Ipc, AvgIpc**: 信息理論描述符

## 3. BCUT 描述符 (BCUT Descriptors)
- **BCUT2D_CHGHI, BCUT2D_CHGLO**: 電荷相關
- **BCUT2D_LOGPHI, BCUT2D_LOGPLOW**: LogP 相關
- **BCUT2D_MRHI, BCUT2D_MRLOW**: 分子折射率相關
- **BCUT2D_MWHI, BCUT2D_MWLOW**: 分子量相關

## 4. 親脂性與極性 (Lipophilicity & Polarity)
- **MolLogP**: LogP (親脂性)
- **MolMR**: 分子折射率
- **TPSA**: 拓撲極性表面積
- **LabuteASA**: Labute 可及表面積
- **HallKierAlpha**: Hall-Kier alpha 值

## 5. 氫鍵相關 (Hydrogen Bonding)
- **NumHDonors**: 氫鍵供體數
- **NumHAcceptors**: 氫鍵受體數
- **NHOHCount**: NH 或 OH 基團數
- **NOCount**: N 或 O 原子數

## 6. 環系統描述符 (Ring System Descriptors)
- **NumAromaticRings**: 芳香環數
- **NumAromaticCarbocycles**: 芳香碳環數
- **NumAromaticHeterocycles**: 芳香雜環數
- **NumSaturatedRings**: 飽和環數
- **NumSaturatedCarbocycles**: 飽和碳環數
- **NumSaturatedHeterocycles**: 飽和雜環數
- **NumAliphaticRings**: 脂肪環數
- **NumAliphaticCarbocycles**: 脂肪碳環數
- **NumAliphaticHeterocycles**: 脂肪雜環數
- **NumHeterocycles**: 雜環數
- **NumBridgeheadAtoms**: 橋頭原子數
- **NumSpiroAtoms**: 螺原子數

## 7. 立體化學描述符 (Stereochemistry)
- **NumAtomStereoCenters**: 立體中心數
- **NumUnspecifiedAtomStereoCenters**: 未指定立體中心數

## 8. 鍵相關描述符 (Bond Descriptors)
- **NumRotatableBonds**: 可旋轉鍵數
- **NumAmideBonds**: 醯胺鍵數

## 9. EState 描述符 (EState Descriptors)
- **MaxEStateIndex, MinEStateIndex**: EState 指數極值
- **MaxAbsEStateIndex, MinAbsEStateIndex**: EState 指數絕對值極值
- **EState_VSA1 到 EState_VSA11**: EState 相關的 VSA 描述符 (11個)
- **VSA_EState1 到 VSA_EState10**: VSA EState 描述符 (10個)

## 10. PEOE 描述符 (PEOE Descriptors)
- **MaxPartialCharge, MinPartialCharge**: 部分電荷極值
- **MaxAbsPartialCharge, MinAbsPartialCharge**: 部分電荷絕對值極值
- **PEOE_VSA1 到 PEOE_VSA14**: PEOE 相關的 VSA 描述符 (14個)

## 11. SMR 描述符 (SMR Descriptors)
- **SMR_VSA1 到 SMR_VSA10**: SMR 相關的 VSA 描述符 (10個)

## 12. SlogP 描述符 (SlogP Descriptors)
- **SlogP_VSA1 到 SlogP_VSA12**: SlogP 相關的 VSA 描述符 (12個)

## 13. 分子指紋密度 (Fingerprint Density)
- **FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3**: Morgan 指紋密度

## 14. 其他描述符
- **SPS**: 結構路徑描述符
- **qed**: 藥物相似性分數 (Quantitative Estimate of Drug-likeness)

## 15. 官能團計數 (Functional Group Counts) - fr_* 系列
共 **85 個官能團描述符**，包括：

### 基本官能團
- **fr_Al_COO, fr_Ar_COO, fr_COO, fr_COO2**: 羧酸基團
- **fr_Al_OH, fr_Ar_OH, fr_Al_OH_noTert**: 羥基
- **fr_C_O, fr_C_O_noCOO**: 羰基
- **fr_C_S**: 碳-硫鍵
- **fr_ether**: 醚
- **fr_ester**: 酯
- **fr_amide, fr_priamide**: 醯胺
- **fr_ketone, fr_ketone_Topliss**: 酮
- **fr_aldehyde**: 醛
- **fr_carboxylic_acid**: 羧酸

### 含氮官能團
- **fr_ArN, fr_Ar_N, fr_Ar_NH**: 芳香氮
- **fr_NH0, fr_NH1, fr_NH2**: 胺基
- **fr_N_O**: 氮-氧鍵
- **fr_aniline**: 苯胺
- **fr_Imine**: 亞胺
- **fr_amidine**: 脒
- **fr_guanido**: 胍基
- **fr_quatN**: 季銨氮
- **fr_azide**: 疊氮
- **fr_azo**: 偶氮
- **fr_diazo**: 重氮
- **fr_nitrile**: 腈
- **fr_isocyan, fr_isothiocyan, fr_thiocyan**: 異氰、異硫氰、硫氰

### 含硫官能團
- **fr_SH**: 巰基
- **fr_sulfide**: 硫醚
- **fr_sulfone**: 碸
- **fr_sulfonamd, fr_prisulfonamd**: 磺醯胺

### 含磷官能團
- **fr_phos_acid**: 磷酸
- **fr_phos_ester**: 磷酸酯

### 含鹵素官能團
- **fr_halogen**: 鹵素
- **fr_alkyl_halide**: 烷基鹵化物

### 含氧官能團
- **fr_epoxide**: 環氧化物
- **fr_oxime**: 肟
- **fr_methoxy**: 甲氧基
- **fr_phenol, fr_phenol_noOrthoHbond**: 酚
- **fr_para_hydroxylation**: 對位羥基化

### 含硝基官能團
- **fr_nitro**: 硝基
- **fr_nitro_arom, fr_nitro_arom_nonortho**: 芳香硝基
- **fr_nitroso**: 亞硝基

### 雜環系統
- **fr_benzene**: 苯環
- **fr_furan**: 呋喃
- **fr_thiophene**: 噻吩
- **fr_pyridine**: 吡啶
- **fr_imidazole**: 咪唑
- **fr_oxazole**: 噁唑
- **fr_thiazole**: 噻唑
- **fr_tetrazole**: 四唑
- **fr_piperdine, fr_piperzine**: 哌啶、哌嗪
- **fr_morpholine**: 嗎啉
- **fr_dihydropyridine**: 二氫吡啶
- **fr_lactam, fr_lactone**: 內醯胺、內酯
- **fr_bicyclic**: 雙環系統
- **fr_benzodiazepine**: 苯並二氮雜
- **fr_barbitur**: 巴比妥
- **fr_Nhpyrrole**: 吡咯氮

### 其他特殊官能團
- **fr_allylic_oxid**: 烯丙基氧化
- **fr_aryl_methyl**: 芳基甲基
- **fr_alkyl_carbamate**: 烷基氨基甲酸酯
- **fr_urea**: 脲
- **fr_term_acetylene**: 末端炔
- **fr_unbrch_alkane**: 無支鏈烷烴
- **fr_HOCCN**: HOCCN 模式
- **fr_Ndealkylation1, fr_Ndealkylation2**: N-去烷基化模式

## 總結

這 217 個描述符涵蓋了：
- **結構特徵**: 原子數、環數、鍵數等
- **拓撲特徵**: 連接性指數、形狀指數等
- **物理化學性質**: LogP、極性表面積、折射率等
- **官能團**: 85 種常見官能團的計數
- **電子性質**: 電荷分布、EState 指數等
- **立體化學**: 立體中心、可旋轉鍵等

這些描述符在藥物性質預測中非常有用，因為它們能夠捕捉分子的多個重要特徵。
