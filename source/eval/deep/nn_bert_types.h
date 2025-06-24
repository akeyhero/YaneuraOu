#ifndef __DLSHOGI_BERT_EVALUATE_H_INCLUDED__
#define __DLSHOGI_BERT_EVALUATE_H_INCLUDED__

#include "../../config.h"
#if defined(YANEURAOU_ENGINE_DEEP_BERT)

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

	// === 入出力の特徴量の定義 (BERT用) ===

	// 各手駒の上限枚数。
	// BERTでは実際の枚数をトークンIDに変換するが、既存コードとの互換性のために定義を残す

	constexpr int MAX_HPAWN_NUM   = 8; // 歩の持ち駒の上限
	constexpr int MAX_HLANCE_NUM  = 4;
	constexpr int MAX_HKNIGHT_NUM = 4;
	constexpr int MAX_HSILVER_NUM = 4;
	constexpr int MAX_HGOLD_NUM   = 4;
	constexpr int MAX_HBISHOP_NUM = 2;
	constexpr int MAX_HROOK_NUM   = 2;

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

	// 先後含めた手駒の枚数の合計
	constexpr u32 MAX_FEATURES2_HAND_NUM = (int)COLOR_NB * MAX_PIECES_IN_HAND_SUM;

	// 駒の種類(成り駒含む。先後の区別はない) : 空の駒は含まないので14種類。
	const int PIECETYPE_NUM = 14;

	// 駒の種類 : 空の駒を含むので15種類。Aperyで定義されている定数
	const int PieceTypeNum = PieceType::DRAGON + 1;

	// 手駒になりうる駒種の数。: Aperyで定義されている定数
	const int HandPieceNum = 7;

	// === BERT特有の定数定義 ===

	// BERTトークン数
	constexpr int BERT_BOARD_LENGTH = SQ_NB;                    // 盤面: 81トークン
	constexpr int BERT_HAND_LENGTH = HandPieceNum * COLOR_NB;   // 持ち駒: 14トークン（手番側7種 + 相手側7種）
	constexpr int BERT_TOTAL_LENGTH = BERT_BOARD_LENGTH + BERT_HAND_LENGTH;  // 合計: 95トークン

	// BERTトークンIDの範囲
	constexpr int BERT_VOCAB_SIZE = 121;  // 0-120のトークンID

	// 盤面のトークンID定義
	constexpr int BERT_EMPTY_ID = 0;           // 空きマス
	constexpr int BERT_BLACK_PIECE_BASE = 1;   // 手番側の駒: 1-8
	constexpr int BERT_BLACK_PROMOTED_BASE = 9; // 手番側の成駒: 9-14
	constexpr int BERT_WHITE_PIECE_BASE = 17;   // 相手側の駒: 17-24
	constexpr int BERT_WHITE_PROMOTED_BASE = 25; // 相手側の成駒: 25-30

	// 持ち駒のトークンID開始位置
	constexpr int BERT_HAND_TOKEN_OFFSET = 31;

	// 持ち駒トークンの各駒種の開始ID
	// 手番側
	constexpr int BERT_BLACK_HAND_PAWN_BASE   = 31;  // 歩: 31-49 (19種類)
	constexpr int BERT_BLACK_HAND_LANCE_BASE  = 50;  // 香: 50-54 (5種類)
	constexpr int BERT_BLACK_HAND_KNIGHT_BASE = 55;  // 桂: 55-59 (5種類)
	constexpr int BERT_BLACK_HAND_SILVER_BASE = 60;  // 銀: 60-64 (5種類)
	constexpr int BERT_BLACK_HAND_GOLD_BASE   = 65;  // 金: 65-69 (5種類)
	constexpr int BERT_BLACK_HAND_BISHOP_BASE = 70;  // 角: 70-72 (3種類)
	constexpr int BERT_BLACK_HAND_ROOK_BASE   = 73;  // 飛: 73-75 (3種類)

	// 相手側
	constexpr int BERT_WHITE_HAND_PAWN_BASE   = 76;  // 歩: 76-94 (19種類)
	constexpr int BERT_WHITE_HAND_LANCE_BASE  = 95;  // 香: 95-99 (5種類)
	constexpr int BERT_WHITE_HAND_KNIGHT_BASE = 100; // 桂: 100-104 (5種類)
	constexpr int BERT_WHITE_HAND_SILVER_BASE = 105; // 銀: 105-109 (5種類)
	constexpr int BERT_WHITE_HAND_GOLD_BASE   = 110; // 金: 110-114 (5種類)
	constexpr int BERT_WHITE_HAND_BISHOP_BASE = 115; // 角: 115-117 (3種類)
	constexpr int BERT_WHITE_HAND_ROOK_BASE   = 118; // 飛: 118-120 (3種類)

	// 移動の定数 (既存との互換性のため定義を残す)
	enum MOVE_DIRECTION {
		UP, UP_LEFT, UP_RIGHT, LEFT, RIGHT, DOWN, DOWN_LEFT, DOWN_RIGHT, UP2_LEFT, UP2_RIGHT,
		UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE,
		MOVE_DIRECTION_NUM,
		MOVE_DIRECTION_NONE = -1
	};

	// 成る移動。10方向。
	const MOVE_DIRECTION MOVE_DIRECTION_PROMOTED[] = {
		UP_PROMOTE, UP_LEFT_PROMOTE, UP_RIGHT_PROMOTE, LEFT_PROMOTE, RIGHT_PROMOTE, DOWN_PROMOTE, DOWN_LEFT_PROMOTE, DOWN_RIGHT_PROMOTE, UP2_LEFT_PROMOTE, UP2_RIGHT_PROMOTE
	};

	// 指し手を表すラベルの数 (BERTでは異なる)
	// BERT: 95（移動元） × 162（移動先: 81マス × 2(成/不成)） = 15,390
	// ただし実際に使用するのは 88（盤面81+手番側持ち駒7） × 162 = 14,256
	constexpr int MAX_MOVE_LABEL_NUM = MOVE_DIRECTION_NUM + HandPieceNum;
	constexpr int BERT_MOVE_FROM_NUM = 95;      // 移動元の総数（盤面81 + 持ち駒14）
	constexpr int BERT_MOVE_TO_NUM = 162;       // 移動先の総数（81マス × 2）
	constexpr int BERT_POLICY_DIM = BERT_MOVE_FROM_NUM * BERT_MOVE_TO_NUM;  // 15,390

	// 特徴量などに使う型
#if defined(TRT_NN_FP16)
	typedef uint8_t PType;
	typedef __half DType;
	extern const DType dtype_zero;
	extern const DType dtype_one;
	inline float to_float(const DType x) {
		return __half2float(x);
	}
	inline DType to_dtype(const float x) {
		return __float2half(x);
	}
#else
	typedef uint8_t PType;
	typedef float DType;
	constexpr const DType dtype_zero = 0.0f;
	constexpr const DType dtype_one  = 1.0f;
	inline float to_float(const DType x) {
		return x;
	}
	inline float to_dtype(const float x) {
		return x;
	}
#endif

	// === BERT用の入力型定義 ===
	// 既存コードとの互換性のため、同じ型名で再定義

	// NNの入力特徴量その1: 盤面トークン列（81要素）
	typedef DType NN_Input1[SQ_NB];

	// NNの入力特徴量その2: 持ち駒トークン列（14要素）
	typedef DType NN_Input2[BERT_HAND_LENGTH];

	// NNの出力特徴量その1 (ValueNetwork) : 期待勝率
	typedef DType NN_Output_Value;

	// NNの出力特徴量その2 (PolicyNetwork) : それぞれの指し手の実現確率
	// BERT: 15,390次元（95 × 162）
	typedef DType NN_Output_Policy[BERT_POLICY_DIM];

	// 入力特徴量を生成する。
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2);

	// 入力特徴量を展開する。
	void extract_input_features(int batch_size, PType* packed_features1, PType* packed_features2, NN_Input1* features1, NN_Input2* features2);

	// 指し手に対して、Policy Networkの返してくる配列のindexを返す。
	int make_move_label(Move move, Color color);

	// Softmax関数
	void softmax_temperature_with_normalize(std::vector<float>& log_probabilities);

	// Softmaxの時のボルツマン温度の設定。
	void set_softmax_temperature(const float temperature);

	// 評価値から価値(勝率)に変換
	float cp_to_value(const Value score , const float eval_coef);

	// 価値(勝率)を評価値[cp]に変換。
	Value value_to_cp(const float score , const float eval_coef);

	// エンジンオプションで設定されたモデルファイル名。
	extern std::vector<std::string> ModelPaths;

} // namespace Eval::dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP_BERT)
#endif // ndef __DLSHOGI_BERT_EVALUATE_H_INCLUDED__