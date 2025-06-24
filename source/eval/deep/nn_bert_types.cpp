#include "nn_bert_types.h"

#if defined(EVAL_DEEP) && defined(YANEURAOU_ENGINE_DEEP_BERT)

#include <cstring> // memset,wchar_t
#include <cmath>   // expf,logf

#include "../../usi.h"
#include "../../bitboard.h"

using namespace std;
using namespace Tools;

namespace
{
	// 指し手に対して、Policy Networkの返してくる配列のindexを返すためのテーブル
	// Eval::init()で初期化する。
	u16 MoveLabel[0x10000][COLOR_NB];
}

namespace Eval::dlshogi
{
	// モデルファイル名へのpath
	std::vector<std::string> ModelPaths;

	// Aperyの手駒は、GOLDが末尾になっていないので変換テーブルを用意する。
	PieceType HandPiece2PieceType[HandPieceNum] = { PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK };

#if defined(TRT_NN_FP16)
	const DType dtype_zero = __float2half(0.0f);
	const DType dtype_one  = __float2half(1.0f);
#endif

	// 盤面の駒をBERTトークンIDに変換
	// position: 局面
	// sq: マス
	// side_to_move: 手番
	PType piece_to_bert_token(Piece pc, Color side_to_move)
	{
		if (pc == NO_PIECE)
			return BERT_EMPTY_ID;

		PieceType pt = type_of(pc);
		Color c = color_of(pc);

		// 手番から見た相対的な色（手番側=BLACK、相手側=WHITE）
		Color relative_color = (c == side_to_move) ? BLACK : WHITE;

		// 基本の駒のID
		int base_id;
		if (relative_color == BLACK) {
			// 手番側の駒
			if (pt <= GOLD) {
				// 成っていない駒: 1-8
				base_id = BERT_BLACK_PIECE_BASE + (pt - PAWN);
			} else {
				// 成駒: 9-14
				// PROM_PAWN(9) → 9, PROM_LANCE(10) → 10, ..., DRAGON(14) → 14
				base_id = BERT_BLACK_PROMOTED_BASE + (pt - PROM_PAWN);
			}
		} else {
			// 相手側の駒
			if (pt <= GOLD) {
				// 成っていない駒: 17-24
				base_id = BERT_WHITE_PIECE_BASE + (pt - PAWN);
			} else {
				// 成駒: 25-30
				base_id = BERT_WHITE_PROMOTED_BASE + (pt - PROM_PAWN);
			}
		}

		return base_id;
	}

	// 持ち駒の枚数をBERTトークンIDに変換
	// hand: 持ち駒
	// pt: 駒種
	// side: 手番から見た相対的な色（BLACK=手番側、WHITE=相手側）
	PType hand_to_bert_token(Hand hand, PieceType pt, Color side)
	{
		int count = hand_count(hand, pt);

		// 各駒種のベースIDを定義
		const int base_ids[COLOR_NB][HandPieceNum] = {
			// 手番側：歩、香、桂、銀、金、角、飛
			{BERT_BLACK_HAND_PAWN_BASE, BERT_BLACK_HAND_LANCE_BASE, BERT_BLACK_HAND_KNIGHT_BASE,
			 BERT_BLACK_HAND_SILVER_BASE, BERT_BLACK_HAND_GOLD_BASE, BERT_BLACK_HAND_BISHOP_BASE,
			 BERT_BLACK_HAND_ROOK_BASE},
			// 相手側
			{BERT_WHITE_HAND_PAWN_BASE, BERT_WHITE_HAND_LANCE_BASE, BERT_WHITE_HAND_KNIGHT_BASE,
			 BERT_WHITE_HAND_SILVER_BASE, BERT_WHITE_HAND_GOLD_BASE, BERT_WHITE_HAND_BISHOP_BASE,
			 BERT_WHITE_HAND_ROOK_BASE}
		};

		// 駒種のインデックス（PAWN=1から始まるので-1）
		int pt_index = pt - PAWN;

		// 最大枚数でクリップ
		const int max_counts[HandPieceNum] = {18, 4, 4, 4, 4, 2, 2}; // 歩は最大18枚
		count = std::min(count, max_counts[pt_index]);

		return base_ids[side][pt_index] + count;
	}

	// 入力特徴量を生成する（BERT版）
	// position: このあとEvalNode()を呼び出したい局面
	// batch_index: バッチ内のインデックス
	// packed_features1: 盤面トークン（81要素）を格納
	// packed_features2: 持ち駒トークン（14要素）を格納
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2)
	{
		// バッチ内の開始位置を計算
		int idx1 = batch_index * SQ_NB;                 // 盤面用（81要素）
		int idx2 = batch_index * BERT_HAND_LENGTH;      // 持ち駒用（14要素）

		Color stm = position.side_to_move();

		// 盤面81トークンをPType配列に格納
		for (Square sq = SQ_11; sq < SQ_NB; ++sq) {
			// 後手番の場合は盤面を180度回転
			Square sq_index = (stm == BLACK) ? sq : Flip(sq);
			Piece pc = position.piece_on(sq);
			packed_features1[idx1 + sq_index] = piece_to_bert_token(pc, stm);
		}

		// 持ち駒14トークンをPType配列に格納
		// 手番側の持ち駒（7種類）
		Hand hand_me = position.hand_of(stm);
		packed_features2[idx2 + 0] = hand_to_bert_token(hand_me, PAWN, BLACK);    // 歩
		packed_features2[idx2 + 1] = hand_to_bert_token(hand_me, LANCE, BLACK);   // 香
		packed_features2[idx2 + 2] = hand_to_bert_token(hand_me, KNIGHT, BLACK);  // 桂
		packed_features2[idx2 + 3] = hand_to_bert_token(hand_me, SILVER, BLACK);  // 銀
		packed_features2[idx2 + 4] = hand_to_bert_token(hand_me, GOLD, BLACK);    // 金
		packed_features2[idx2 + 5] = hand_to_bert_token(hand_me, BISHOP, BLACK);  // 角
		packed_features2[idx2 + 6] = hand_to_bert_token(hand_me, ROOK, BLACK);    // 飛

		// 相手側の持ち駒（7種類）
		Hand hand_opp = position.hand_of(~stm);
		packed_features2[idx2 + 7]  = hand_to_bert_token(hand_opp, PAWN, WHITE);   // 歩
		packed_features2[idx2 + 8]  = hand_to_bert_token(hand_opp, LANCE, WHITE);  // 香
		packed_features2[idx2 + 9]  = hand_to_bert_token(hand_opp, KNIGHT, WHITE); // 桂
		packed_features2[idx2 + 10] = hand_to_bert_token(hand_opp, SILVER, WHITE); // 銀
		packed_features2[idx2 + 11] = hand_to_bert_token(hand_opp, GOLD, WHITE);   // 金
		packed_features2[idx2 + 12] = hand_to_bert_token(hand_opp, BISHOP, WHITE); // 角
		packed_features2[idx2 + 13] = hand_to_bert_token(hand_opp, ROOK, WHITE);   // 飛
	}

	// 入力特徴量を展開する（BERT版）
	// BERTではトークンIDをそのまま使用するため、PType(uint8_t)からDType(float)への変換のみ
	void extract_input_features(int batch_size, PType* packed_features1, PType* packed_features2, NN_Input1* features1, NN_Input2* features2)
	{
		// PType配列をNN_Input1, NN_Input2の配列に変換
		// 注：BERTでは各NN_Input1/2は1次元配列なので、batch_sizeの数だけ配列がある
		for (int b = 0; b < batch_size; ++b) {
			// 盤面81トークン
			for (int i = 0; i < SQ_NB; ++i) {
				features1[b][i] = to_dtype(static_cast<float>(packed_features1[b * SQ_NB + i]));
			}

			// 持ち駒14トークン
			for (int i = 0; i < BERT_HAND_LENGTH; ++i) {
				features2[b][i] = to_dtype(static_cast<float>(packed_features2[b * BERT_HAND_LENGTH + i]));
			}
		}
	}

	// MoveLabel配列を事前に初期化する（BERT版）
	// "isready"に対して呼び出される。
	void init_move_label()
	{
		// BERTのmake_move_label
		auto bert_make_move_label = [](Move16 move16, Color color)
		{
			Square to = move16.to_sq();
			Square from = move16.from_sq();
			bool drop = move16.is_drop();
			bool promote = move16.is_promote();

			// 後手の場合、盤面を180度回転
			if (color == WHITE) {
				to = Flip(to);
				from = Flip(from);
			}

			// 移動元インデックス（0-94）
			int from_index;
			if (!drop) {
				// 盤上の移動: 0-80
				from_index = from;
			} else {
				// 駒打ち: 81-87（手番側の持ち駒のみ）
				PieceType pt = move16.move_dropped_piece();
				from_index = 81 + int(pt - PAWN);
			}

			// 移動先インデックス（0-161）
			// to: 0-80, promote時は81を加算
			int to_index = to + (promote ? 81 : 0);

			// 最終的なラベル = from × 162 + to
			return from_index * BERT_MOVE_TO_NUM + to_index;
		};

		for (auto color : COLOR)
			// 手駒はfromの位置がSQ_NB～SQ_NB+6
			for(Square from_sq = SQ_ZERO; from_sq < SQ_NB + 7; ++from_sq)
				for (auto to_sq : SQ)
					// 成りと成らずと
					for (int promote = 0; promote < 2; ++promote)
					{
						// 駒打ちであるか
						bool drop = from_sq >= SQ_NB;

						// 駒打ちの成りはない
						if (drop && promote)
							continue;

						Move16 move;
						if (!drop)
						{
							move = !promote
								? make_move16(from_sq, to_sq)
								: make_move_promote16(from_sq, to_sq);
						}
						else {
							PieceType pt = (PieceType)(from_sq - (int)SQ_NB + PAWN);
							move = make_move_drop16(pt, to_sq);
						}

						// BERTのmake_move_label()を呼び出して初期化する。
						MoveLabel[move.to_u16()][color] = bert_make_move_label(move, color);
					}
	}

	// 指し手に対して、Policy Networkの返してくる配列のindexを返す。
	int make_move_label(Move move, Color color)
	{
		return MoveLabel[move.to_u16()][color];
	}

	// Softmaxの時の温度パラメーター。
	constexpr float default_softmax_temperature = 1.0f;
	float beta = 1.0f / default_softmax_temperature;

	void set_softmax_temperature(const float temperature) {
		beta = 1.0f / temperature;
	}

	void softmax_temperature_with_normalize(std::vector<float> &log_probabilities) {
		// apply beta exponent to probabilities(in log space)
		float max = numeric_limits<float>::min();
		for (float& x : log_probabilities) {
			x *= beta;
			if (x > max) {
				max = x;
			}
		}

		// オーバーフローを防止するため最大値で引く
		float sum = 0.0f;
		for (float& x : log_probabilities) {
			x = expf(x - max);
			sum += x;
		}

		// normalize
		const float scale = 1.0f / sum;
		for (float& x : log_probabilities) {
			x *= scale;
		}
	}

	// 評価値から価値(勝率)に変換
	float cp_to_value(const Value score, const float eval_coef)
	{
		// 勝率を[0,1]の値に変換する
		//   score :  評価値
		//   eval_coef : 勝率を評価値に変換する時の定数。default = 756
		// 返し値 : 勝率∈[0,1]
		return 1.0f / (1.0f + expf(-(float)score / eval_coef));
	}

	// 価値(勝率)を評価値[cp]に変換
	Value value_to_cp(const float score, const float eval_coef)
	{
		// 勝率を評価値[cp]に変換する。
		//   score : 勝率∈[0,1]
		//   eval_coef : 勝率を評価値に変換する時の定数。default = 756
		// 返し値 : 評価値
		if (score == 1.0f)
			return VALUE_MATE;
		else if (score == 0.0f)
			return -VALUE_MATE;
		else
			return (Value)(-logf(1.0f / score - 1.0f) * eval_coef);
	}

} // namespace Eval::dlshogi

#endif // defined(EVAL_DEEP) && defined(YANEURAOU_ENGINE_DEEP_BERT)