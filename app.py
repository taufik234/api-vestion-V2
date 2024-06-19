import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# Load the model in the SavedModel format

#try:
# Define the directory where the model is stored
model_dir = "model/MyModel.h5"

# Define the filename of the model
model_filename = "MyModel.h5"

# Construct the full path to the model file by joining the directory and filename
model_path = "model/MyModel.h5"

# Load the TensorFlow Keras model from the specified file path
model = tf.keras.models.load_model(model_path)

print("Model loaded successfully.")
#except Exception as e:
    #print(f"Error loading model: {e}")

        


# Class names and vege information
class_names = ['Brokoli Hijau', 'Daun Pepaya', 'Daun Singkong', 'Daun Kelor', 'Kembang Kol', 'Kubis Hijau ', 'Paprika Merah', 'Sawi sendok atau Pakcoy', 'Tomat Merah', 'Wortel Nantes']
vege_info = {
    'Brokoli Hijau': {
        'nama_latin': "Brassica oleracea var. italica",
        'deskripsi': "Brokoli ini berasal dari daerah Laut Tengah (Italy), banyak dijumpai di daerah beriklim sedang hingga sejuk. Biasanya mudah dijumpai di pasar dan  memiliki rasa yang sedikit pahit dan teksturnya renyah tetapi bisa dimakan mentah. Warna brokoli Italia hijau memiliki warna hijau tua pada kuntumnya. Batangnya juga berwarna hijau, meskipun bisa sedikit lebih pucat dibandingkan kuntumnya. Kuntum brokoli tersusun rapat dan berbentuk seperti payung yang mengelompok di bagian atas batang utama. Batang brokoli tebal dan berwarna hijau. Bagian bawah batang biasanya lebih keras, sementara bagian atas batang lebih lunak dan lezat untuk dimakan.",
        'kalori': "34 kal /100 gram",
        'karbohidrat': "6,6gr /100 gram",
        'protein': "2,8gr /100 gram",
        'lemak': "0,3gr /100 gram",
        'serat': "2,6gr /100 gram",
        'vitamin': "Vitamin C, B4, E, B5, B6, B2, K, A, B6",
        'mineral': "Folat, Mangan, Kalium, Fosfor, Calsium, Natrium, Magnesium, Zat Besi, Zinc",
        'manfaat': "Brokoli hijau dapat meningkatkan sistem kekebalan tubuh seperti mencegah kanker karena mengandung senyawa sulforafan, memelihara kesehatan jantung, kesehatan mata, kesehatan tulang dan gigi karena mengandung vitamin K, menurunkan berat badan, menjaga kesehatan pencernaan, meningkatkan kesehatan kulit",
        'pemilihan': "Untuk menentukan pemilihan brokoli yang bagus bisa dilihat dari kepala bunga padat berwarna hijau tua, batang kokoh, dan daun hijau serta hindari kepala bunga lembek, kuning, atau layu. Coba tekan batang bagian atas dengan ujung jari, karena yang masih segar umumnya punya bagian batang yang cenderung masih keras dan tidak lunak. Juga coba tekan pangkal batang, jika terasa lunak dan cenderung berair, bisa jadi indikator hampir busuk.",
        'penyimpanan_jangka_pendek': [ 
            "1. Simpan di kulkas: Brokoli segar utuh akan bertahan sekitar 5-7 hari di kulkas.",
            "2. Potong-Potong: Potong brokoli menjadi kuntum yang lebih kecil sebelum disimpan dapat mengurangi waktu penyimpanan menjadi 3-4 hari.",
            "3. Bungkus dengan plastik: Pastikan brokoli dibungkus rapat dengan plastik kedap udara untuk mencegah kelembapan berlebih dan terkontaminasi oleh bau makanan lain di kulkas.",
            "4. Jauhkan dari etilena: Etilena adalah gas yang dikeluarkan oleh beberapa buah dan sayuran yang dapat mempercepat pembusukan brokoli (Jauhkan brokoli dari buah-buahan seperti apel,pisang, dan melon)."    
        ],
        'penyimpanan_jangka_panjang': [
            "1. Blanching: Blanching adalah metode dengan merebus brokoli dalam air mendidih selama 1-2 menit kemudian rendam ke dalam air es selama 3 menit.",
            "2. Bekukan: Brokoli yang sudah diblanching kemudian dikeringkan dan dibekukan, Brokoli beku dapat bertahan hingga 12 bulan.",
            "3. Pengeringan: Brokoli dapat dikeringkan dengan oven atau dehidrator, Brokoli kering dapat disimpan pada wadah kedap udara di tempat yang sejuk dan gelap hingga 1 tahun."
        ],
        'menus': [
            {
                "nama_menu": "Brokoli Siram Telur",
                "menu_url": "https://endeus.tv/resep/brokoli-siram-telur"
            },
            {
                "nama_menu": "Telur Dadar Kaya Gizi",
                "menu_url": "https://www.masakapahariini.com/resep/cara-membuat-telur-dadar-untuk-mpasi/"
            }
            ,
            {
                "nama_menu": "Kari Sayur Brokoli",
                "menu_url": "https://cookpad.com/id/resep/11158430-kari-sayur-brokoli"
            }
        ]
    },
    'Kembang Kol': {
        'nama_latin': "Brassica oleracea var. botrytis",
        'deskripsi': "Kembang kol berasal dari daerah Laut Tengah, termasuk dari variasi warna dari brokoli Italia hijau (Brassica oleracea var. italica). Warna putih disebabkan oleh faktor genetik yang menghasilkan pigmen antosianin dalam jumlah yang lebih sedikit dibandingkan brokoli hijau. Memiliki rasa yang lebih manis dan teksturnya lebih lembut dibandingkan brokoli Italia hijau dan tentunnya bisa dimakan mentah.",
        'kalori': "25 kal /100 gram",
        'karbohidrat': "5,3gr /100 gram",
        'protein': "1,92gr /100 gram",
        'lemak': "0,1gr /100 gram",
        'serat': "2,5gr /100 gram",
        'vitamin': "Vitamin A, B1, B2, B3, C, E, K",
        'mineral': "Kalsium, Zat besi, Magnesium, Mangan, Zinc, Tembaga, fosfor, kalium",
        'manfaat': "Kembang kol dapat meningkatkan sistem kekebalan tubuh seperti mencegah kanker karena mengandung senyawa sulforafan, kesehatan jantung, kesehatan mata, kesehatan tulang karena mengandung vitamin K, menurunkan berat badan, menjaga kesehatan pencernaan, karena mengandung kolin dapat mencegah penumpukan kolesterol dalam pembuluh darah.",
        'pemilihan': "Untuk menentukan pemilihan kembang kol yang bagus bisa dilihat dari kepala bunga padat berwarna putih atau krem cerah, batang kokoh, dan daun berwarna hijau, serta hindari kepala bunga lembek, menguning, layu, dan berbintik hitam. Gunakan indra penciuman untuk mengenali ciri-ciri kebusukan, serta aroma kurang sedap dan cenderung asam. Tekan batang bagian atas dengan ujung jari sebab yang masih segar umumnya punya bagian batang yang cenderung masih keras dan tidak lunak. Amati pangkal batang dan coba tekan bagian tersebut, jika terasa lunak dan cenderung berair, bisa jadi indikator hampir busuk.",
        'penyimpanan_jangka_pendek': [
            "1. Jangan cuci: Jangan mencuci kembang kol saat akan disimpan, cuci jika hanya saat akan digunakan.",
            "2. Bungkus: Bungkus seluruh kepala kembang kol dalam kantong berlubang atau dengan kertas (menggunakan kertas korang tidak dianjurkan) dan masukkan ke dalam lemari es (cara ini dapat membuat kuntum kembang kol tetap segar dalam 5 hari)."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Bersihkan: Bersihkan kembang kol dengan air bersih mengalir.",
            "2. Potong: Potong kuntum kembang kol menjadi bagian yang kecil.",
            "3. Blanching: Blanching adalah metode dengan merebus kembang kol dalam air mendidih selama 1-2 menit kemudian rendam ke dalam air es selama 3 menit.",
            "4.Keringkan dan bekukan:  Kembang kol yang sudah diblanching kemudian dikeringkan dan dibekukan, Kembang kol beku dapat bertahan hingga 12 bulan.",
            "5. Pengeringan: Kembang kol dapat dikeringkan dengan oven atau dehidrator, Kembang kol kering dapat disimpan pada wadah kedap udara di tempat yang sejuk dan gelap hingga 1 tahun."
        ],
        'menus': [
            {
                "nama_menu": "Sayur Santan Kembang Kol",
                "menu_url": "https://cookpad.com/id/resep/23840512-sayur-santan-kembang-kol"
            },
            {
                "nama_menu": "Nasi Goreng Kembang Kol",
                "menu_url": "https://cookpad.com/id/resep/23838866-nasi-goreng-sehat-kembang-kol"
            },
            {
                "nama_menu": "Kembang Kol Goreng Tepung",
                "menu_url": "https://cookpad.com/id/resep/22972269-kembang-kol-goreng-tepung"
            }
        ]
    },
    'Sawi sendok atau Pakcoy': {
        'nama_latin': "Brassica rapa subsp. chinensis",
        "deskripsi": "Sawi sendok atau pakcoy berasal dari Tiongkok (China), di mana jenis sayuran ini telah menjadi bagian integral dari masakan Tionghoa tradisional selama berabad-abad. Tumbuh subur di iklim dingin dengan curah hujan yang cukup dan tumbuh paling baik di tempat teduh parsial dan membutuhkan penyiraman secara teratur. Sawi sendok dapat dipanen dalam waktu 40-50 hari setelah tanam. Sawi sendok, dikenal juga sebagai pakcoy memiliki daun hijau yang memiliki daun yang lebar dan tangkai yang panjang dan memiliki rasa yang segar dan renyah serta bisa dimakan mentah.",
        'kalori': "21 kal /100 gram",
        'karbohidrat': "3,9gr /100 gram",
        'protein': "1,8gr /100 gram",
        'lemak': "0,3gr /100 gram",
        'serat': "0,7gr /100 gram",
        'vitamin': "vitamin A, B1, B2 dan C",
        'mineral': "Kalium, Fosfor, Zat besi, Natrium, Kalium, Kalsium",
        'manfaat': "Sawi sendok, menawarkan banyak keuntungan untuk kesehatan. Kandungan vitamin C-nya meningkatkan daya tahan tubuh, seratnya melancarkan pencernaan, dan senyawa glikosinolatnya membantu mencegah kanker. Sawi sendok juga kaya kalium untuk tekanan darah stabil, kalsium dan vitamin K untuk tulang kuat, lutein dan zeaxanthin untuk kesehatan mata, dan membantu menurunkan berat badan. Konsumsi sawi sendok secara rutin untuk tubuh yang lebih sehat.",
        'pemilihan': "Untuk menentukan pemilihan pakcoy yang bagus, perhatikan daun yang berwarna hijau cerah dan mengkilap, tidak layu atau menguning, bebas dari lubang, bintik hitam, atau kerusakan, serta memiliki tekstur yang renyah dan tidak lembek. Batangnya harus berwarna putih atau hijau muda, kokoh dan tidak mudah patah, serta ukurannya tidak terlalu besar. Anda dapat mencium bau pakcoy untuk memastikan kesegarannya; pakcoy segar memiliki aroma yang segar dan tidak berbau busuk. Anda juga dapat menekan batang pakcoy untuk memastikan kekencangannya; batang pakcoy segar harus terasa kokoh dan tidak lembek. Secara keseluruhan, sawi sendok yang bagus terlihat segar dan tidak layu, bebas dari pestisida dan kotoran, serta memiliki aroma yang segar.",
        'penyimpanan_jangka_pendek': [
            "1. Di Kulkas: Simpan sawi sendok di dalam kulkas, bungkus dengan kain lembab atau tisu kertas berlubang untuk menjaga kelembapannya tanpa membuatnya terlalu basah dan masukkan kedalam kantong plastik berlubang atau kantong penyimpanan sayuran yang bisa menjaga kelembapan.",
            "2. Tanpa Dicuci: Sawi sendok sebaiknya tidak dicuci sebelum disimpan karena kelembapan berlebih dapat mempercepat pembusukan, Cucilah sawi hanya saat akan digunakan.",
            "3. Laci Sayuran: Simpan sawi sendok di laci sayuran kulkas yang biasanya memiliki suhu dan kelembapan yang tepat untuk sayuran hijau."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Pembekuan: Sawi sendok bisa dibekukan setelah melalui proses blanching, Celupkan sawi dalam air mendidih selama 1-2 menit kemudian segera masukkan ke dalam air es untuk menghentikan proses pemasakan dan tiriskan-keringkan-simpan dalam kantong kedap udara sebelum dimasukkan ke dalam freezer.",
            "2. Pengeringan: Potong sawi sendok menjadi bagian yang lebih kecil kemudian keringkan dengan sinar matahari atau menggunakan alat pengering (dehydrator) hingga benar-benar kering, simpan dalam wadah kedap udara di tempat yang sejuk dan kering.",
            "3. Pengawetan: Sawi sendok bisa diawetkan dengan cara difermentasi atau diasinkan."
        ],
        'menus': [
            {
                "nama_menu": "Pakcoy Bawang Putih",
                "menu_url": "https://cookpad.com/id/resep/22658054-pakcoy-bawang-putih-ala-resto?ref=search&search_term=pakcoy"
            },
            {
                "nama_menu": "Tumis Tahu Putih dan Sawi Pakcoy",
                "menu_url": "https://cookpad.com/id/resep/23840224-tumis-tahu-putih-dan-sawi-pokcoy"
            },
            {
                "nama_menu": "Oseng Pakcoy Tahu",
                "menu_url": "https://cookpad.com/id/resep/22972199-oseng-pokcoy-oyong-tahu"
            }
        ]
    },
    'Daun Pepaya': {
        'nama_latin': "Carica papaya L.",
        'deskripsi': "Daun pepaya berasal dari tanaman pepaya (Carica papaya), yang aslinya berasal dari Amerika Tengah dan Meksiko. Daun-daun ini biasanya berbentuk seperti telapak tangan dengan daun-daun kecil yang tersusun secara spiral pada tangkai tengahnya. Mereka memiliki rasa yang pahit dimana rasa pait pada daun pepaya disebabkan oleh senyawa kimia yang disebut dengan karpain. Daun pepaya juga mengandung alkaloid karpain yang dalam jumlah besar dapat bersifat toksik, oleh karena itu, daun pepaya biasanya dimasak atau diolah terlebih dahulu.",
        'kalori': "79 kal /100 gram",
        'karbohidrat': "11,9gr /100 gram",
        'protein': "8gr /100 gram",
        'lemak': "2gr /100 gram",
        'serat': "1,5gr /100 gram",
        'vitamin': "vitamin A, B1, B2, B3, Beta karoten, C, dan K",
        'mineral': "Zat besi, kalium, Natrium, seng",
        'manfaat': "Daun pepaya memiliki sejumlah manfaat kesehatan yang menarik. Mereka kaya akan enzim papain yang dapat membantu dalam pencernaan makanan, mengurangi peradangan, dan mempercepat penyembuhan luka. Mengontrol kadar gula darah, menambah kelancaran ASI,  membantu proses penyembuhan demam berdarah dengue, konsumsi daun pepaya secara teratur juga dapat membantu menurunkan risiko penyakit jantung, meningkatkan produksi sel darah merah, dan memperbaiki kesehatan pencernaan.",
        'pemilihan': "Untuk menentukan pemilihan daun pepaya yang bagus untuk dimasak, pilih daun yang masih muda dan berwarna hijau cerah, karena daun yang lebih tua cenderung memiliki rasa pahit yang kuat. Pilih daun dengan ukuran yang cukup besar dan tidak terlalu kecil, karena daun pepaya yang besar cenderung lebih muda dan segar. Pastikan daun bersih dan tidak terlalu kotor, serta bebas dari tanda-tanda jamur atau penyakit lainnya.",
        'penyimpanan_jangka_pendek': [
            "1. Cuci bersih: Cuci bersih daun pepaya dan keringkan dengan lembut.",
            "2. Letakkan pada wadah: Letakkan daun dalam kantong plastik atau wadah kedap udara yang dilapisi dengan tisu basah.",
            "3. Simpan pada suhu dingin: Simpan di bagian kulkas yang lebih dingin."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Rebus daun: Daun pepaya yang sudah direbus perlu direndam dalam air dingin terlebih dahulu, hal ini bertujuan untuk menghentikan proses pematangannya.",
            "2. Peras: Lalu peras daun pepaya agar kandungan airnya berkurang dan kepal-kepal dan bagi menjadi beberapa bagian.",
            "3. siapkan wadah: Siapkan plastik dan masukkan satu per satu kepalan daun pepaya sebelumnya.",
            "4. Simpan pada freezer: Daun pepaya yang sudah dimasukkan ke plastik, bisa diletakkan di freezer (bisa tahan 1 bulan)."
        ],
        'menus': [
            {
                "nama_menu": "Tumis Daun Pepaya Teri Jengki",
                "menu_url": "https://cookpad.com/id/resep/22675350-tumis-daun-pepaya-teri-jengki?ref=search&search_term=daun%20pepaya"
            },
            {
                "nama_menu": "Oseng Daun Pepaya",
                "menu_url": "https://cookpad.com/id/resep/22604460-oseng-daun-pepaya"
            },
            {
                "nama_menu": "Sayur Daun Pepaya Kuah Santan",
                "menu_url": "https://cookpad.com/id/resep/17281014-sayur-daun-pepaya-kuah-santan"
            }
        ]
    },
    'Daun Singkong': {
        'nama_latin': "Manihot esculanta crantz",
        'deskripsi': "Daun singkong merupakan  tanaman asli Amerika Tengah dan Selatan. Tanaman singkong dikenal tahan terhadap kondisi tanah yang kurang subur dan cuaca yang kering, sehingga cocok untuk tumbuh di daerah tropis dan subtropis. Daun singkong memiliki rasa yang agak pahit dan tekstur yang cukup keras jika dimakan mentah. Daun singkong mengandung glikosida sianogenik, yang dapat menghasilkan sianida, Oleh karena itu, daun singkong tidak boleh dimakan mentah.",
        'kalori': "31 kal /100 gram",
        'karbohidrat': "5,9gr /100 gram",
        'protein': "3,4gr /100 gram",
        'lemak': "0,5gr /100 gram",
        'serat': "4,3gr /100 gram",
        'vitamin': "vitamin A, B1, B2, B3, C",
        'mineral': "Kalium, Tembaga, Zat Besi, Seng",
        'manfaat': "Daun singkong memiliki beragam manfaat kesehatan yang menarik. Daun singkong  mengandung senyawa fitokimia yang memiliki sifat antioksidan dan antiinflamasi, yang dapat membantu melawan radikal bebas dalam tubuh dan mengurangi risiko peradangan. Konsumsi daun singkong secara teratur juga dikaitkan dengan peningkatan sistem kekebalan tubuh, penurunan risiko penyakit jantung, dan menjaga kesehatan mata dan kulit. Selain itu, daun singkong juga dapat membantu meningkatkan pencernaan, mengurangi risiko sembelit, dan membantu mengontrol kadar gula darah.",
        'pemilihan': "Untuk menentukan pemilihan daun singkong yang bagus, jangan memilih daun yang masih muda (pucuk daun), pilihlah yang 3-4 tangkai ke bawah. Pilih daun yang berwarna hijau tua dan cerah, serta hindari daun yang terlalu tua atau yang sudah menguning. Pilih daun yang ukurannya cukup besar dan tidak terlalu kecil. Sentuh daun untuk memastikan teksturnya kaku dan tidak layu, hindari daun yang terlalu lembek atau layu. Hindari juga daun yang rusak, sobek, atau berlubang karena bisa menjadi tanda bahwa daun tersebut tidak segar atau telah terkena hama.",
        'penyimpanan_jangka_pendek': [
            "1. Bungkus dengan kresek: Daun singkong tidak usah dicuci tetapi langsung masukan kedalam kresek (plastik) daun singkong.",
            "2. Bungkus: Bungkus kresek (plastik) kemudian plester atau bisa menggunakan alat lain agar udara tidak masuk.",
            "3. Simpan: Masukan dalam rak kulkas paling bawah."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Rebus daun (opsional): Daun singkong yang sudah direbus perlu direndam dalam air dingin terlebih dahulu. Hal ini bertujuan untuk menghentikan proses pematangannya.",
            "2. Peras (jika merebus daun): Lalu peras daun singkong agar kandungan airnya berkurang dan kepal-kepal dan bagi menjadi beberapa bagian.",
            "3. Siapkan wadah: Jika merebus daun masukkan satu per satu kepalan daun singkong sebelumnya dan jika tidak merebus daun masukkan daun ke dalam plastik-keluarkan udara dalam plastik-ikat.",
            "4. Simpan pada freezer: Daun singkong yang sudah dimasukkan ke plastik, bisa diletakkan di freezer."
        ],
        'menus': [
            {
                "nama_menu": "Sayur Daun Singkong Kuah Santan",
                "menu_url": "https://cookpad.com/id/resep/22607477-sayur-daun-singkong-kuah-santan"
            },
            {
                "nama_menu": "Gulai Daun Singkong",
                "menu_url": "https://cookpad.com/id/resep/22656120-gulai-daun-singkong"
            },
            {
                "nama_menu": "Oseng Daun Singkong Teri",
                "menu_url": "https://cookpad.com/id/resep/22553928-oseng-daun-singkong-teri"
            }
        ]
    },
    'Daun Kelor': {
        'nama_latin': "Moringa oleifera",
        'deskripsi': "Daun kelor atau merunggai berasal dari wilayah India utara. Memiliki kandungan gizi yang sangat tinggi dan sering dianggap sebagai \"superfood.\". Daun-daunya kecil, berwarna hijau tua, dan berbentuk bulat atau oval, memiliki rasa yang ringan dan khas",
        'kalori': "9,2 kal /100 gram",
        'karbohidrat': "12,5gr /100 gram",
        'protein': "6,7gr /100 gram",
        'lemak': "1,7gr /100 gram",
        'serat': "0,9gr /100 gram",
        'vitamin': "vitamin A, B1(Thiamin), B2 (Riboflavin), B3 (Niasin), B6, C, E",
        'mineral': "Kalsium, Kalium, Magnesium, Fosfor, Zat besi",
        'manfaat': "Daun kelor (Moringa oleifera) dikenal sebagai superfood karena kandungan nutrisinya yang luar biasa. Daun kelor dapat menjaga tekanan darah karena kaya akan kalium, menurunkan kolesterol tinggi, kaya antioksidan, menurunkan resiko terkena penyakit jantung koroner dan stroke, baik untuk penderita diabetes, mengurangi peradangan atau inflamasi, menjaga  daya tahan tubuh karena kaya akan vitamin C, menjaga kesehatan mata, mencegah anemia, Menjaga kesehatan gigi dan tulang.",
        'pemilihan': "Untuk menentukan pemilihan daun kelor yang bagus, pilih daun yang berwarna hijau tua dan segar, hindari daun yang berwarna kuning atau coklat, karena ini bisa menjadi tanda bahwa daun tersebut sudah tua (rasanya pahit) atau rusak. Pilih daun yang memiliki tekstur yang segar dan renyah, karena daun yang layu atau lembek mungkin sudah tidak segar lagi. Daun kelor yang segar memiliki aroma yang khas dan segar, jadi hindari daun yang memiliki bau yang tidak sedap atau aneh. Pastikan daun kelor bersih dan bebas dari kotoran, serangga, atau sisa-sisa pestisida, dan cuci daun dengan baik sebelum digunakan.",
        'penyimpanan_jangka_pendek': [
            "1. Siapkan wadah: Siapkan wadah yang natinya bisa ditutup dan letakkan tisu pada dasar wadah (agar daun tetap kering).",
            "2. Ambil daun: Pisahkan daun kelor dengan tangkainya dan letakkan dalam wadah yang sudah dilapisi dengan tisu.",
            "3. Tutup: Jika sudah terkumpul semua lapisi lagi bagian atas dengan tisu, sebisa mungkin menutupi permukaan dan tutup.",
            "4. Letakkan di kulkas: Letakkan pada kulkas (Dapat bertahan 5 hari sampai 1 minggu)."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Cuci: Cuci daun kelor yang sudah dipisahkan dengan tangkainya.",
            "2. Letakkan pada nampan: Sebarkan daun di atas nampan yang dilapisi kertas perkamen dalam satu lapisan.",
            "3. Bekukan: Bekukan daun selama beberapa jam hingga benar-benar beku.",
            "4. Pindah: Pindahkan daun beku ke dalam kantong freezer atau wadah kedap udara.",
            "5. Simpan: Simpan di freezer hingga beberapa bulan."
        ],
        'menus': [
            {
                "nama_menu": "Sayur Labu Air Daun Kelor",
                "menu_url": "https://cookpad.com/id/resep/22618367-sayur-labu-air-daun-kelor"
            },
            {
                "nama_menu": "Bakwan Wortel Daun Kelor",
                "menu_url": "https://cookpad.com/id/resep/22657790-bakwan-wortel-dan-daun-kelor"
            },
            {
                "nama_menu": "Omlete Kornet Daun Kelor",
                "menu_url": "https://cookpad.com/id/resep/22587760-omlete-kornet-daun-kelor"
            }
        ]
    },
    'Kubis Hijau': {
        'nama_latin': "Brassica oleracea var. capitata.",
        'deskripsi': "Kubis hijau berasal dari daerah Mediterania dan Eropa Selatan. Tanaman ini memiliki daun yang tebal, renyah, dan berlapis-lapis yang membentuk kepala yang padat dan bulat. Warna daunnya bervariasi dari hijau muda hingga hijau tua, dengan permukaan daun yang halus dan bertekstur. Daun kubis hijau memiliki rasa yang sedikit manis dan renyah ketika dimakan mentah. Untuk mengonsumsinya mentah, pastikan untuk mencuci kubis dengan baik untuk menghilangkan kotoran dan residu pestisida.",
        'kalori': "25 kal /100 gram",
        'karbohidrat': "5,8gr /100 gram",
        'protein': "1,8gr /100 gram",
        'lemak': "0,1gr /100 gram",
        'serat': "2,50mg /100 gram",
        'vitamin': "Vitamin A, C, K, E, B1, B3, B5, B6",
        'mineral': "Calcium, Zat besi, Magnesium, Fosfor, Potassium, Zinc, Tembaga, Mangan, Selenium",
        'manfaat': "Kubis hijau ternyata menyimpan segudang manfaat untuk kesehatan. Kaya akan serat, kubis hijau membantu melancarkan pencernaan dan mencegah sembelit. Vitamin C di dalamnya berperan penting dalam meningkatkan sistem kekebalan tubuh dan melawan infeksi. Tak hanya itu, kubis hijau juga kaya akan vitamin K1 yang membantu menjaga kesehatan jantung dengan mencegah pengapuran arteri. Senyawa antioksidan dalam kubis hijau pun membantu melindungi tubuh dari kanker, dan vitamin K1 serta mangannya penting untuk kesehatan tulang dan membantu mencegah osteoporosis.",
        'pemilihan': "Untuk menentukan pemilihan kubis hijau yang bagus, pilih kubis yang berwarna hijau cerah dan tidak pucat, serta hindari kubis yang berwarna kuning, layu, atau berbintik coklat. Bentuk kubis yang ideal adalah bulat dan padat, hindari kubis berbentuk tidak teratur. Pilih kubis dengan daun yang segar dan tidak layu, tekan daun dengan jari anda; daun yang segar akan terasa kenyal dan tidak lembek. Hindari kubis yang daunnya keras atau bertekstur kasar, karena daun ini biasanya sudah tua dan pahit. Pilih kubis dengan batang yang masih segar dan berwarna hijau, hindari batang yang layu atau berwarna coklat. Batang yang tebal menandakan daun yang matang dan memiliki rasa yang lebih kuat.",
        'penyimpanan_jangka_pendek': [
            "1. Simpan: Simpan kubis hijau di dalam kulkas pada suhu dingin, letakkan kubis di dalam laci crisper di bagian bawah kulkas untuk menjaga kelembapannya, hindari menyimpan kubis di dekat buah-buahan seperti apel dan pisang karena gas etilen yang dilepaskan oleh buah-buahan ini dapat mempercepat pembusukan kubis.",
            "2. Jangan cuci: Jangan mencuci kubis hijau sebelum disimpan karena air dapat mempercepat pembusukan, cucilah hanya sebelum digunakan.",
            "3. Bungkus: Bungkus kubis hijau dengan kertas handuk basah untuk menjaga kelembapannya.",
            "4. Tips Tambahan: Gunakan wadah kedap udara untuk menyimpan potongan kubis hijau yang sudah dipotong, potong kubis hijau 2-3 hari sebelum digunakan agar lebih tahan lama."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Potong dan rebus: Potong kubis hijau menjadi potongan kecil dan rebus potongan kubis selama 1-2 menit untuk melunakkan teksturnya.",
            "2. Dingingkan: Tiriskan dan dinginkan potongan kubis.",
            "3. Masukkan dalam wadah: Kemas potongan kubis dalam kantong plastik kedap udara atau wadah freezer.",
            "4. Pembekuan: Bekukan kubis dalam freezer dengan suhu -18°C (0°F) atau lebih rendah."
        ],
        'menus': [
            {
                "nama_menu": "Oseng Kubis Hijau",
                "menu_url": "https://cookpad.com/id/resep/13351956-oseng-kubis-ijo-klothokan"
            },
            {
                "nama_menu": "Kubis Hijau Dengan Saus Tiram",
                "menu_url": "https://combionline.com/id/resep/24390/kubis-hijau-dengan-saus-tiram"
            },
            {
                "nama_menu": "Kubis Buncis Bumbu Hijau",
                "menu_url": "https://cookpad.com/id/resep/17249657-kubis-buncis-bumbu-hijau"
            }
        ]
    },
    'Wortel Nantes': {
        'nama_latin': "Daucus carota L. ssp. sativus var. Nantes.",
        'deskripsi': "Wortel Nantes diperkirakan berasal dari daerah Nantes di Prancis. Warna oranye cerah dan merata, warna oranye wortel Nantes berasal dari pigmen alami yang disebut beta karoten. Kulit halus dan tipis. Bentuknya yang panjang dan ramping, serta rasanya yang manis dan sedikit earthy.",
        'kalori': "41 kal /100 gram",
        'karbohidrat': "9,6gr /100 gram",
        'protein': "0,9gr /100 gram",
        'lemak': "0,2gr /100 gram",
        'serat': "2,8gr /100 gram",
        'vitamin': "Vitamin A, B1 (Thiamin), B2 (Riboflavin), B3 (Niasin), Vitamin C, Vitamin B6,  Vitamin K1",
        'mineral': "Kalsium, Fosfor, Zat besi, Natrium, Kalium, Tembaga, Zinc",
        'manfaat': "Wortel Nantes kaya akan nutrisi yang sangat bermanfaat bagi kesehatan, dimana manfaatnya yaitu sebagai antioksidan salah satunya Beta Karoten (dimasak terlebih dahulu akan memaksimalkan penyerapan pada tubuh dibanding mengonsumsi secara mentah), selain Beta Karoten terdapat Alfa Karoten, Lutein (bagus untuk kesehatan mata). terdapat juga Lycopene dan antosianin serta Polyacetylenes yang sangat bagus untuk sel darah merah. Mengonsumsi wortel dapat mengurangi resiko cancer, Menurunkan kolesterol darah, Membantu menurunkan berat badan, Menjaga kesehatan mata.",
        'pemilihan': "Untuk menentukan pemilihan wortel Nantes yang bagus, pilih wortel dengan warna oranye cerah dan merata, hindari wortel yang berwarna pucat atau kusam, karena ini menandakan kurang matang atau sudah tua. Pilih wortel dengan kulit yang halus dan tidak memar, karena kulit yang memar atau retak dapat menandakan kerusakan atau infeksi. Bentuk wortel yang ideal adalah lurus dan ramping, hindari wortel yang bengkok atau bercabang, karena ini menandakan pertumbuhan yang tidak normal. Pilih wortel dengan ukuran sedang, sekitar 15-20 cm, karena wortel yang terlalu kecil mungkin kurang matang, sedangkan wortel yang terlalu besar mungkin sudah tua dan berserat.",
        'penyimpanan_jangka_pendek': [
            "1. Simpan di kulkas: Tempatkan wortel Nantes di dalam kantong plastik kedap udara atau wadah kedap udara yang dibawahnya dilapisi tisu dan letakkan di laci sayuran kulkas.",
            "2. Cuci sebelum digunakan: Cuci wortel Nantes hanya sebelum digunakan untuk mencegah kelembaban yang berlebihan.",
            "3. Jauhkan dari buah: Hindari menyimpan wortel Nantes di dekat buah-buahan seperti apel dan pir karena gas etilen yang dilepaskan oleh buah-buahan dapat mempercepat pembusukan wortel (dapat bertahan 1-2 minggu)."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Cuci: Cuci bersih wortel dan keringkan dari air.",
            "2. Kupas: Kupas kulit luar wortel dan potong pangkal atas wortel.",
            "3. Potong: potong potong wortel sesuai kebutuhan.",
            "4. Masukkan wadah: Masukkan wortel ke dalam wadah dan tutup rapat. ",
            "5. Simpan: Simpan wortel didalam freezer (dapat bertahan sampai 5 bulan)."
        ],
        'menus': [
            {
                "nama_menu": "Orak Arik Wortel",
                "menu_url": "https://cookpad.com/id/resep/22660996-orak-arik-wortel-sawi-putih"
            },
            {
                "nama_menu": "Nasi Goreng Wortel Saos Tiram",
                "menu_url": "https://cookpad.com/id/resep/22595326-707-nasi-goreng-wortel-saos-tiram"
            },
            {
                "nama_menu": "Sup Wortel",
                "menu_url": "https://cookpad.com/id/resep/22675964-sup-wortel"
            }
        ]
    },
    'Tomat Merah': {
        'nama_latin': "Solanum lycopersicum syn. Lycopersicum esculentum",
        'deskripsi': "Tomat merah adalah tumbuhan dari keluarga Solanaceae, tumbuhan asli Amerika Tengah dan Selatan, dari Meksiko sampai Peru. Bentuk tomat merah umumnya berbentuk bulat atau lonjong. Warna merah ini berasal dari pigmen likopen yang terkandung di dalamnya. Daging tomat merah berwarna merah cerah atau merah tua. Rasanya manis dan sedikit asam dan bisa langsung dimakan. Daging tomat mengandung banyak air dan biji.",
        'kalori': "25 kal /100 gram",
        'karbohidrat': "3,9gr /100 gram",
        'protein': "0,9gr /100 gram",
        'lemak': "0,2gr  /100 gram",
        'serat': "1,2gr /100 gram",
        'vitamin': "Vitamin A, K, B1, B3, B5, B6, C",
        'mineral': "Folat, Kalium, Zat besi, Magnesium, Kromium, Kolin, Seng, dan Fosfor",
        'manfaat': "Tomat merah, si buah bulat berwarna cerah menyimpan segudang manfaat kesehatan. Mengandung Likopen yang mana antioksidan. Dapat membantu kesehatan jantung karena mengandung Likopen yang dapat membantu menurunkan kolesterol LDL (jahat), memberikan efek perlindungan pada lapisan dalam pembuluh darah dan dapat menurunkan risiko pembekuan darah. Dapat membantu pencegahan kanker termasuk kanker payudara, dapat menjaga kesehatan kulit",
        'pemilihan': "Untuk menentukan pemilihan tomat merah yang bagus, pilih tomat dengan warna merah merata dan cerah, hindari tomat yang kusam atau memiliki bintik hitam karena warna merah yang cerah menandakan tomat matang sempurna. Pilih tomat dengan kulit halus, mulus, dan tidak berkerut, serta hindari tomat yang memar, berlubang, atau retak. Jika masih ada batangnya, pilih batang yang masih hijau dan menempel kuat pada tomat; batang yang kering atau terlepas menandakan tomat sudah tua. Pilih tomat dengan bentuk yang bulat atau lonjong simetris, hindari tomat yang bentuknya tidak teratur atau cacat.",
        'penyimpanan_jangka_pendek': [
            "1. Simpan di kulkas: Bungkus tomat rapat dengan plastik atau wadah kedap udara, letakkan di rak paling bawah kulkas dan hindari area dekat pintu kulkas karena fluktuasi suhu dapat mempercepat layu.",
            "2. Cuci sebelum konsumsi: Mencuci tomat terlebih dahulu dapat mempercepat pembusukan."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Iris tomat: Iris tomat menjadi irisan tipis.",
            "2. Keringkan: Tempatkan di atas rak pengering atau dehidrator dan keringkan hingga benar-benar kering.",
            "3. Simpan: Simpan tomat kering dalam wadah kedap udara di tempat yang sejuk dan gelap."
        ],
        'menus': [
            {
                "nama_menu": "Telur Orek Tomat",
                "menu_url": "https://cookpad.com/id/resep/22592954-telur-orek-tomat"
            },
            {
                "nama_menu": "Sup Oyong Tomat Telur",
                "menu_url": "https://cookpad.com/id/resep/22615789-sop-oyong-tomat-telur-menu-simpel-satset"
            },
            {
                "nama_menu": "Telur Ceplok Saos Kecap Tomat",
                "menu_url": "https://cookpad.com/id/resep/22669522-telor-ceplok-saos-kecap-tomat"
            }
        ]
    },
    'Paprika Merah': {
        'nama_latin': "Capsicum annuum L.",
        'deskripsi': "Paprika merupakan tanaman asli dari Amerika Tengah dan Amerika Selatan. Warna merah cerah yang mencolok. Warnanya berasal dari kandungan likopen yang tinggi. Paprika merah memiliki rasa manis yang khas dan sedikit pahit di bagian bijinya. Rasa manisnya lebih dominan dibandingkan paprika hijau dan kuning. Rasa manis ini berasal dari kandungan karotenoid yang lebih tinggi pada paprika merah.",
        'kalori': "27 kal /100 gram",
        'karbohidrat': "6,65gr /100 gram",
        'protein': "0,9gr /100 gram",
        'lemak': "0,13gr /100 gram",
        'serat': "1,2gr /100 gram",
        'vitamin': "Vitamin A, C, B6",
        'mineral': "Kalsium, Zat besi, Kalium",
        'manfaat': "Warnanya yang merah dan memiliki beberapa manfaat diantaranya, paprika merupakan sumber antioksidan, mendukung kesehatan mata, menurunkan resiko penyakit jantung, menyokong sistem kekebalan tubuh, mendukung kesehatan pencernaan.",
        'pemilihan': "Untuk menentukan pemilihan paprika merah yang bagus, pilihlah paprika yang berwarna cerah dan merata, karena warna yang cerah menunjukkan bahwa paprika masih segar dan matang sempurna. Hindari paprika yang memiliki banyak bintik coklat besar atau bintik-bintik hitam, karena bintik-bintik hitam menandakan bahwa paprika sudah busuk. Pilihlah paprika yang memiliki kulit halus dan mengkilap, karena ini menandakan bahwa paprika masih segar, serta hindari paprika yang memiliki kulit keriput atau berkerut, karena memar pada paprika dapat menandakan kerusakan pada bagian dalamnya. Pilihlah paprika yang memiliki batang masih hijau dan segar, karena batang yang hijau dan segar menandakan bahwa paprika baru dipetik. Hindari paprika dengan batang coklat atau layu, karena ini menandakan bahwa paprika sudah tua atau tidak segar. Pilihlah paprika yang terasa keras dan padat saat ditekan, dan hindari paprika yang terasa lunak atau lembek saat ditekan, karena ini menandakan bahwa paprika sudah tua atau tidak segar.",
        'penyimpanan_jangka_pendek': [
            "1. Simpan di Lemari Es: Letakkan paprika merah yang belum dicuci di dalam kantong plastik atau wadah kedap udara dan didalamnya beri tisu atau paper towel, simpan di laci sayuran di lemari es karena paprika merah dapat bertahan selama 1 hingga 2 minggu dengan cara ini.",
            "2. Jauhkan dari Buah-Buahan yang Menghasilkan Etilen: Hindari menyimpan paprika merah bersama dengan buah-buahan seperti apel pisang atau tomat karena buah-buahan tersebut menghasilkan etilen yang bisa mempercepat pematangan dan pembusukan paprika."
        ],
        'penyimpanan_jangka_panjang': [
            "1. Cuci dan keringkan: Cuci dan keringkan paprika merah dengan baik lalu potong paprika merah sesuai kebutuhan (misalnya, potong dadu atau iris tipis memanjang).",
            "2. Bekukan: Letakkan potongan paprika tersebut kedalam nampan yang dialasi kertas roti dan bekukan selama kurang lebih 6 jam/sampai beku.",
            "3. Masukkan ke wadah: Letakkan potongan paprika dalam kantong plastik khusus freezer atau wadah kedap udara dan simpan di freezer (Paprika dapat bertahan hingga 10-12 bulan)."
        ],
        'menus': [
            {
                "nama_menu": "Tumis Ikan Asin Paprika Merah",
                "menu_url": "https://cookpad.com/id/resep/14426719-142-tumis-ikan-asin-paprika-merah"
            },
            {
                "nama_menu": "Chicken Teriyaki Paprika Merah",
                "menu_url": "https://cookpad.com/id/resep/17064501-chicken-teriyaki-paprika-merah"
            },
            {
                "nama_menu": "Brokoli Cah Paprika Merah",
                "menu_url": "https://cookpad.com/id/resep/16537491-brokoli-cah-paprika-merah"
            }
        ]
    }
}

class_names = list(vege_info.keys())

def preprocess_image(img_path):
    """
    Preprocess the input image to fit the model input requirements.

    Parameters:
    - img_path: Path to the image file.

    Returns:
    - img_array: Preprocessed image array.
    """
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Image data not found'}), 400

        image_file = request.files['image']
        image_data = io.BytesIO(image_file.read())
        img_array = preprocess_image(image_data)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0][predicted_class]
        is_above_threshold = bool(confidence > 0.5)  # Convert to regular Python boolean

        # Get vege info
        predicted_vege = class_names[predicted_class]
        vege_details = vege_info[predicted_vege]

        # Return prediction result
        return jsonify({
            'message': 'Model is predicted successfully.',
            'data': {
                'result': predicted_vege,
                'confidenceScore': float(confidence * 100),  # Convert to percentage
                'isAboveThreshold': is_above_threshold,
                'nama_latin': vege_details['nama_latin'],
                'deskripsi': vege_details['deskripsi'],
                'kalori': vege_details['kalori'],
                'karbohidrat': vege_details['karbohidrat'],
                'protein': vege_details['protein'],
                'lemak': vege_details['lemak'],
                'serat': vege_details['serat'],
                'vitamin': vege_details['vitamin'],
                'mineral': vege_details['mineral'],
                'manfaat': vege_details['manfaat'],
                'pemilihan': vege_details['pemilihan'],
                'penyimpanan_jangka_pendek': vege_details['penyimpanan_jangka_pendek'],
                'penyimpanan_jangka_panjang': vege_details['penyimpanan_jangka_panjang'],
                'menus': vege_details['menus'],
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))