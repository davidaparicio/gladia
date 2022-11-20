from typing import Dict

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

TASK = "translation"
CKPT = "facebook/nllb-200-distilled-600M"


# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
codes_as_string = """ace	Acehnese (Arabic script)	ace_Arab
ace	Acehnese (Latin script)	ace_Latn
acm	Mesopotamian Arabic	acm_Arab
acq	Ta’izzi-Adeni Arabic	acq_Arab
aeb	Tunisian Arabic	aeb_Arab
afr	Afrikaans	afr_Latn
ajp	South Levantine Arabic	ajp_Arab
aka	Akan	aka_Latn
amh	Amharic	amh_Ethi
apc	North Levantine Arabic	apc_Arab
arb	Modern Standard Arabic	arb_Arab
arb	Modern Standard Arabic (Romanized)	arb_Latn
ars	Najdi Arabic	ars_Arab
ary	Moroccan Arabic	ary_Arab
arz	Egyptian Arabic	arz_Arab
asm	Assamese	asm_Beng
ast	Asturian	ast_Latn
awa	Awadhi	awa_Deva
ayr	Central Aymara	ayr_Latn
azb	South Azerbaijani	azb_Arab
azj	North Azerbaijani	azj_Latn
bak	bak Bashkir	bak_Cyrl
bam	bam Bambara	bam_Latn
ban	ban Balinese	ban_Latn
bel	bel Belarusian	bel_Cyrl
bem	bem Bemba	bem_Latn
ben	Bengali	ben_Beng
bho	Bhojpuri	bho_Deva
bjn	Banjar (Arabic script)	bjn_Arab
bjn	Banjar (Latin script)	bjn_Latn
bod	Standard Tibetan	bod_Tibt
bos	Bosnian	bos_Latn
bug	Buginese	bug_Latn
bul	Bulgarian	bul_Cyrl
cat	Catalan	cat_Latn
ceb	Cebuano	ceb_Latn
ces	Czech	ces_Latn
cjk	Chokwe	cjk_Latn
ckb	Central Kurdish	ckb_Arab
crh	Crimean Tatar	crh_Latn
cym	Welsh	cym_Latn
dan	Danish	dan_Latn
deu	German	deu_Latn
dik	Southwestern Dinka	dik_Latn
dyu	Dyula	dyu_Latn
dzo	Dzongkha	dzo_Tibt
ell	Greek	ell_Grek
eng	English	eng_Latn
epo	Esperanto	epo_Latn
est	Estonian	est_Latn
eus	Basque	eus_Latn
ewe	Ewe	ewe_Latn
fao	Faroese	fao_Latn
fij	Fijian	fij_Latn
fin	Finnish	fin_Latn
fon	Fon	fon_Latn
fra	French	fra_Latn
fur	Friulian	fur_Latn
fuv	Nigerian Fulfulde	fuv_Latn
gla	Scottish Gaelic	gla_Latn
gle	Irish	gle_Latn
glg	Galician	glg_Latn
grn	Guarani	grn_Latn
guj	Gujarati	guj_Gujr
hat	Haitian Creole	hat_Latn
hau	Hausa	hau_Latn
heb	Hebrew	heb_Hebr
hin	Hindi	hin_Deva
hne	Chhattisgarhi	hne_Deva
hrv	Croatian	hrv_Latn
hun	Hungarian	hun_Latn
hye	Armenian	hye_Armn
ibo	Igbo	ibo_Latn
ilo	Ilocano	ilo_Latn
ind	Indonesian	ind_Latn
isl	Icelandic	isl_Latn
ita	Italian	ita_Latn
jav	Javanese	jav_Latn
jpn	Japanese	jpn_Jpan
kab	Kabyle	kab_Latn
kac	Jingpho	kac_Latn
kem	Kamba	kam_Latn
ken	Kannada	kan_Knda
kas	Kashmiri (Arabic script)	kas_Arab
kas	Kashmiri (Devanagari script)	kas_Deva
kat	Georgian	kat_Geor
knc	Central Kanuri (Arabic script)	knc_Arab
knc	Central Kanuri (Latin script)	knc_Latn
kaz	Kazakh	kaz_Cyrl
kbp	Kabiyè	kbp_Latn
kea	Kabuverdianu	kea_Latn
khm	Khmer	khm_Khmr
kik	Kikuyu	kik_Latn
kin	Kinyarwanda	kin_Latn
kir	Kyrgyz	kir_Cyrl
kmb	Kimbundu	kmb_Latn
kmr	Northern Kurdish	kmr_Latn
kon	Kikongo	kon_Latn
kor	Korean	kor_Hang
lao	Lao	lao_Laoo
lij	Ligurian	lij_Latn
lim	Limburgish	lim_Latn
lin	Lingala	lin_Latn
lit	Lithuanian	lit_Latn
lmo	Lombard	lmo_Latn
ltg	Latgalian	ltg_Latn
ltz	Luxembourgish	ltz_Latn
lua	Luba-Kasai	lua_Latn
lug	Ganda	lug_Latn
luo	Luo	luo_Latn
lus	Mizo	lus_Latn
lvs	Standard Latvian	lvs_Latn
mag	Magahi	mag_Deva
mai	Maithili	mai_Deva
mal	Malayalam	mal_Mlym
mar	Marathi	mar_Deva
min	Minangkabau (Arabic script)	min_Arab
min	Minangkabau (Latin script)	min_Latn
mkd	Macedonian	mkd_Cyrl
plt	Plateau Malagasy	plt_Latn
mlt	Maltese	mlt_Latn
mni	Meitei (Bengali script)	mni_Beng
khk	Halh Mongolian	khk_Cyrl
mos	Mossi	mos_Latn
mri	Maori	mri_Latn
mya	Burmese	mya_Mymr
nld	Dutch	nld_Latn
nno	Norwegian Nynorsk	nno_Latn
nob	Norwegian Bokmål	nob_Latn
npi	Nepali	npi_Deva
nso	Northern Sotho	nso_Latn
nus	Nuer	nus_Latn
nyw	Nyanja	nya_Latn
oci	Occitan	oci_Latn
gaz	West Central Oromo	gaz_Latn
ory	Odia	ory_Orya
pag	Pangasinan	pag_Latn
pan	Eastern Panjabi	pan_Guru
pap	Papiamento	pap_Latn
pes	Western Persian	pes_Arab
pol	Polish	pol_Latn
por	Portuguese	por_Latn
prs	Dari	prs_Arab
pbt	Southern Pashto	pbt_Arab
quy	Ayacucho Quechua	quy_Latn
ron	Romanian	ron_Latn
run	Rundi	run_Latn
rus	Russian	rus_Cyrl
sag	Sango	sag_Latn
san	Sanskrit	san_Deva
sat	Santali	sat_Olck
scn	Sicilian	scn_Latn
shn	Shan	shn_Mymr
sin	Sinhala	sin_Sinh
slk	Slovak	slk_Latn
slv	Slovenian	slv_Latn
smo	Samoan	smo_Latn
sna	Shona	sna_Latn
snd	Sindhi	snd_Arab
som	Somali	som_Latn
sot	Southern Sotho	sot_Latn
spa	Spanish	spa_Latn
als	Tosk Albanian	als_Latn
srd	Sardinian	srd_Latn
srp	Serbian	srp_Cyrl
ssw	Swati	ssw_Latn
sun	Sundanese	sun_Latn
swe	Swedish	swe_Latn
swh	Swahili	swh_Latn
szl	Silesian	szl_Latn
tam	Tamil	tam_Taml
tat	Tatar	tat_Cyrl
tel	Telugu	tel_Telu
tgk	Tajik	tgk_Cyrl
tgl	Tagalog	tgl_Latn
tha	Thai	tha_Thai
tir	Tigrinya	tir_Ethi
taq	Tamasheq (Latin script)	taq_Latn
taq	Tamasheq (Tifinagh script)	taq_Tfng
tpi	Tok Pisin	tpi_Latn
tsn	Tswana	tsn_Latn
tso	Tsonga	tso_Latn
tuk	Turkmen	tuk_Latn
tum	Tumbuka	tum_Latn
tur	Turkish	tur_Latn
twi	Twi	twi_Latn
tzm	Central Atlas Tamazight	tzm_Tfng
uig	Uyghur	uig_Arab
ukr	Ukrainian	ukr_Cyrl
umb	Umbundu	umb_Latn
urd	Urdu	urd_Arab
uzn	Northern Uzbek	uzn_Latn
vec	Venetian	vec_Latn
vie	Vietnamese	vie_Latn
war	Waray	war_Latn
wol	Wolof	wol_Latn
xho	Xhosa	xho_Latn
ydd	Eastern Yiddish	ydd_Hebr
yor	Yoruba	yor_Latn
yue	Yue Chinese	yue_Hant
zho	Chinese (Simplified)	zho_Hans
zho	Chinese (Traditional)	zho_Hant
zsm	Standard Malay	zsm_Latn
zul	zu	zul_Latn"""

codes_as_string = codes_as_string.split("\n")

flores_codes_mapping = {}
for code in codes_as_string:
    iso_3, lang, lang_code = code.split("\t")
    flores_codes_mapping[iso_3] = lang_code


def predict(
    input_string: str, source_language: str, target_language: str
) -> Dict[str, str]:
    """
    Translate the text from source lang to target lang

    Args:
        input_string (str): the text to translate
        source_language (str): the language of the text 3-letters ISO code representation of the source language to translate from
        target_language (str): the language to translate to 3-letters ISO code representation of the source language to translate from

    Returns:
        str: the translated text
    """
    max_length = 400

    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT)
    tokenizer = AutoTokenizer.from_pretrained(CKPT)

    # mapping ISO CODE 3 letter (ISO 639-3) to flores 200 code
    # https://github.com/facebookresearch/flores/blob/main/flores200/README.md
    flores200_source_language = flores_codes_mapping[source_language]
    flores200_target_language = flores_codes_mapping[target_language]

    translation_pipeline = pipeline(
        TASK,
        model=model,
        tokenizer=tokenizer,
        src_lang=flores200_source_language,
        tgt_lang=flores200_target_language,
        max_length=max_length,
    )

    result = translation_pipeline(input_string)
    return {
        "prediction": result[0]["translation_text"],
        "prediction_raw": result[0]["translation_text"],
    }
