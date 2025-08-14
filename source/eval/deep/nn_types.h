#ifndef __DLSHOGI_EVALUATE_H_INCLUDED__
#define __DLSHOGI_EVALUATE_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../position.h"

#if defined(TENSOR_RT)
//#define TRT_NN_FP16
#define UNPACK_CUDA
#endif

#if defined(TRT_NN_FP16)
#include <cuda_fp16.h>
#endif

namespace Eval::dlshogi
{
	// === GPU関連の設定 ===

	// GPUの最大数(これ以上のGPUは扱えない)
#if !defined(MAX_GPU)
	constexpr int MAX_GPU = 16;
#endif
	constexpr int max_gpu = MAX_GPU;

	// === 入出力の特徴量の定義 ===

	// 各手駒の上限枚数。

	constexpr int MAX_HPAWN_NUM   = 18; // 歩の持ち駒の上限
	constexpr int MAX_HLANCE_NUM  =  4;
	constexpr int MAX_HKNIGHT_NUM =  4;
	constexpr int MAX_HSILVER_NUM =  4;
	constexpr int MAX_HGOLD_NUM   =  4;
	constexpr int MAX_HBISHOP_NUM =  2;
	constexpr int MAX_HROOK_NUM   =  2;

	// AperyのHandPiece enumの順が、やねうら王のPieceType順と異なるので、
	// このテーブルを参照するときには順番に注意。
	// ※　歩、香、桂、銀、金、角、飛の順。
	const int MAX_PIECES_IN_HAND[] =
	{
		MAX_HPAWN_NUM   , // PAWN
		MAX_HLANCE_NUM  , // LANCE
		MAX_HKNIGHT_NUM , // KNIGHT
		MAX_HSILVER_NUM , // SILVER
		MAX_HGOLD_NUM   , // GOLD
		MAX_HBISHOP_NUM , // BISHOP
		MAX_HROOK_NUM   , // ROOK
	};

	// 手駒の枚数の合計
	constexpr u32 MAX_PIECES_IN_HAND_SUM = MAX_HPAWN_NUM + MAX_HLANCE_NUM + MAX_HKNIGHT_NUM + MAX_HSILVER_NUM + MAX_HGOLD_NUM + MAX_HBISHOP_NUM + MAX_HROOK_NUM;

	// 駒の種類(成り駒含む。先後の区別はない) : 空の駒は含まないので14種類。
	const int PIECETYPE_NUM = 14;

	// 駒の種類 : 空の駒を含むので15種類。Aperyで定義されている定数
	const int PieceTypeNum = PieceType::DRAGON + 1;

	// 手駒になりうる駒種の数。: Aperyで定義されている定数
	const int HandPieceNum = 7;

	// === BERT特有の定数定義 ===

	// BERTトークン数
	constexpr int BERT_BOARD_TOKEN_NUM = SQ_NB;                        // 盤面: 81トークン
	constexpr int BERT_HAND_TOKEN_NUM = HandPieceNum * (int)COLOR_NB;  // 持ち駒: 14トークン（手番側7種 + 相手側7種）
	constexpr int BERT_TOTAL_TOKEN_NUM = BERT_BOARD_TOKEN_NUM + BERT_HAND_TOKEN_NUM;  // 合計: 95トークン

	// BERTトークンIDの範囲
	constexpr int BERT_VOCAB_SIZE = 121;  // 0-120のトークンID

	// 盤面のトークンID定義
	constexpr int BERT_EMPTY_ID            =  0; // 空きマス
	constexpr int BERT_BLACK_PIECE_BASE    =  1; // 手番側の駒: 1-8
	constexpr int BERT_BLACK_PROMOTED_BASE =  9; // 手番側の成駒: 9-14
	constexpr int BERT_WHITE_PIECE_BASE    = 17; // 相手側の駒: 17-24
	constexpr int BERT_WHITE_PROMOTED_BASE = 25; // 相手側の成駒: 25-30

	// 持ち駒のトークンID開始位置
	constexpr int BERT_HAND_TOKEN_OFFSET = 31;

	// 持ち駒トークンの各駒種の開始ID
	// 手番側
	constexpr int BERT_BLACK_HAND_PAWN_BASE   = 31; // 歩: 31-49 (19種類)
	constexpr int BERT_BLACK_HAND_LANCE_BASE  = 50; // 香: 50-54 (5種類)
	constexpr int BERT_BLACK_HAND_KNIGHT_BASE = 55; // 桂: 55-59 (5種類)
	constexpr int BERT_BLACK_HAND_SILVER_BASE = 60; // 銀: 60-64 (5種類)
	constexpr int BERT_BLACK_HAND_GOLD_BASE   = 65; // 金: 65-69 (5種類)
	constexpr int BERT_BLACK_HAND_BISHOP_BASE = 70; // 角: 70-72 (3種類)
	constexpr int BERT_BLACK_HAND_ROOK_BASE   = 73; // 飛: 73-75 (3種類)

	// 相手側
	constexpr int BERT_WHITE_HAND_PAWN_BASE   =  76; // 歩: 76-94 (19種類)
	constexpr int BERT_WHITE_HAND_LANCE_BASE  =  95; // 香: 95-99 (5種類)
	constexpr int BERT_WHITE_HAND_KNIGHT_BASE = 100; // 桂: 100-104 (5種類)
	constexpr int BERT_WHITE_HAND_SILVER_BASE = 105; // 銀: 105-109 (5種類)
	constexpr int BERT_WHITE_HAND_GOLD_BASE   = 110; // 金: 110-114 (5種類)
	constexpr int BERT_WHITE_HAND_BISHOP_BASE = 115; // 角: 115-117 (3種類)
	constexpr int BERT_WHITE_HAND_ROOK_BASE   = 118; // 飛: 118-120 (3種類)

	// 各升の数と手駒の駒種の合計
	// Transformer にトークンとして入力する
	constexpr u32 MAX_FEATURES1_NUM = BERT_BOARD_TOKEN_NUM/*盤上の駒*/ + BERT_HAND_TOKEN_NUM/*手駒*/;

	// トークンではない入力の特徴量を想定
	// 現在未使用
	constexpr u32 MAX_FEATURES2_NUM = 0;

	// 指し手を表すラベルの数
	// この数(95×2)×升の数(SQ_NB=81)だけPolicy Networkが値を出力する。
	// ×2 は、成りと不成の2通り。
	// 駒の順はAperyのPieceTypeの順。これは、やねうら王と同じ。
	// ※　歩、香、桂、銀、角、飛、金。
	constexpr int MAX_MOVE_LABEL_NUM = (int)MAX_FEATURES1_NUM * 2;

	// 特徴量などに使う型。
	//
	// 16bitが使えるのは、cuDNNのときだけだが、cuDNNの利用はdlshogiでは廃止する予定らしいので、
	// ここでは32bit floatとして扱う。
#if defined(TRT_NN_FP16)
	typedef uint8_t PType;
	typedef __half DType;
	inline float to_float(const DType x) {
		return __half2float(x);
	}
	inline DType to_dtype(const float x) {
		return __float2half(x);
	}
#else
	typedef uint8_t PType;
	typedef float DType;
	inline float to_float(const DType x) {
		return x;
	}
	inline float to_dtype(const float x) {
		return x;
	}
#endif

	// NNの入力特徴量その1
	// ※　dlshogiでは、features1_tという型名。
	typedef DType NN_Input1[MAX_FEATURES1_NUM];

	// NNの入力特徴量その2
	// ※　dlshogiでは、features2_tという型名。
	typedef DType NN_Input2[MAX_FEATURES2_NUM];

	// NNの出力特徴量その1 (ValueNetwork) : 期待勝率
	typedef DType NN_Output_Value;

	// NNの出力特徴量その2 (PolicyNetwork) : それぞれの指し手の実現確率
	typedef DType NN_Output_Policy[MAX_MOVE_LABEL_NUM * int(SQ_NB)];

	// 入力特徴量を生成する。
	//   position  : このあとEvalNode()を呼び出したい局面
	//   features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2);

	// 入力特徴量を展開する。GPU側で展開する場合は不要。
	void extract_input_features(int batch_size, PType* packed_features1, PType* packed_features2, NN_Input1* features1, NN_Input2* features2);

	// 指し手に対して、Policy Networkの返してくる配列のindexを返す。
	int make_move_label(Move move, Color color);

	// Softmax関数
	void softmax_temperature_with_normalize(std::vector<float>& log_probabilities);

	// Softmaxの時のボルツマン温度の設定。
	void set_softmax_temperature(const float temperature);

	// "isready"に対して設定されたNNのModel Pathを取得する。
	//std::vector<std::string> GetModelPath();

	// 評価値から価値(勝率)に変換
	// スケールパラメータは、elmo_for_learnの勝率から調査した値
	// 何かの変換の時に必要になる。
	float cp_to_value(const Value score , const float eval_coef);

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	// 	 eval_coef : 勝率を評価値に変換する時の定数。default = 756
	//
	// 返し値 :
	//   +29900は、評価値の最大値
	//   -29900は、評価値の最小値
	//   +30000,-30000は、(おそらく)詰みのスコア
	Value value_to_cp(const float score , const float eval_coef);

	// エンジンオプションで設定されたモデルファイル名。(フォルダ名含む)
	// このsize() == max_gpuのはず。
	// "isready"で初期化されている。
	extern std::vector<std::string> ModelPaths;

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)
#endif // ndef __DLSHOGI_EVALUATE_H_INCLUDED__

