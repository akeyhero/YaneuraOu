#include "nn_types.h"

#if defined(EVAL_DEEP)

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
	PieceType HandPiece2PieceType[HandPieceNum ] = { PAWN , LANCE , KNIGHT , SILVER , GOLD , BISHOP , ROOK };
	int       PieceType2HandPiece[PIECE_TYPE_NB] = { -1 , 0 , 1 , 2 , 3 , 5 , 6 , 4 }; // [1] がPAWN

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
				// PRO_PAWN(9) → 9, PRO_LANCE(10) → 10, ..., DRAGON(14) → 14
				base_id = BERT_BLACK_PROMOTED_BASE + (pt - PRO_PAWN);
			}
		} else {
			// 相手側の駒
			if (pt <= GOLD) {
				// 成っていない駒: 17-24
				base_id = BERT_WHITE_PIECE_BASE + (pt - PAWN);
			} else {
				// 成駒: 25-30
				base_id = BERT_WHITE_PROMOTED_BASE + (pt - PRO_PAWN);
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
	//   position: このあとEvalNode()を呼び出したい局面
	//   batch_index: バッチ内のインデックス
	//   packed_features1 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	//   packed_features2 : ここに書き出す。(事前に呼び出し元でバッファを確保しておくこと)
	void make_input_features(const Position& position, int batch_index, PType* packed_features1, PType* packed_features2)
	{
		int idx_board = batch_index * MAX_FEATURES1_NUM;
		int idx_hand = idx_board + (int)SQ_NB;

		Color stm = position.side_to_move();

		// 盤面81トークンをPType配列に格納
		for (Square sq = SQ_11; sq < SQ_NB; ++sq) {
			// 後手番の場合は盤面を180度回転
			Square sq_index = (stm == BLACK) ? sq : Flip(sq);
			Piece pc = position.piece_on(sq);
			packed_features1[idx_board + sq_index] = piece_to_bert_token(pc, stm);
		}

		// 持ち駒14トークンをPType配列に格納
		Hand hand_me = position.hand_of(stm);
		Hand hand_opp = position.hand_of(~stm);
		for (int i = 0; i < HandPieceNum; ++i) {
			packed_features1[idx_hand + i    ] = hand_to_bert_token(hand_me,  HandPiece2PieceType[i], BLACK);
			packed_features1[idx_hand + i + 7] = hand_to_bert_token(hand_opp, HandPiece2PieceType[i], WHITE);
		}
	}

	// 入力特徴量を展開する（BERT版）
	// BERTではトークンIDをそのまま使用するため、PType[]からNNInput[][]へのアラインメント変換のみ
	void extract_input_features(int batch_size, PType* packed_features1, PType* packed_features2, NN_Input1* features1, NN_Input2* features2)
	{
		// PType配列をNN_Input1, NN_Input2の配列に変換
		for (int b = 0; b < batch_size; ++b) {
			// 盤面81トークン
			for (int i = 0; i < MAX_FEATURES1_NUM; ++i) {
				features1[b][i] = packed_features1[b * MAX_FEATURES1_NUM + i];
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
				from_index = int(SQ_NB) + PieceType2HandPiece[pt];
			}

			// 移動先インデックス（0-161）
			// to: 0-80, promote時は81を加算
			int to_index = to + (promote ? 81 : 0);

			// 最終的なラベル = from × 162 + to
			return from_index * (BERT_BOARD_TOKEN_NUM * 2) + to_index;
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

	// Boltzmann distribution
	// see: Reinforcement Learning : An Introduction 2.3.SOFTMAX ACTION SELECTION
	// →　第二版が無料で公開されているので、そちらを参照するようにしたほうが良いのでは。
	//		Reinforcement Learning: An Introduction 2nd Ed.
	//		http://incompleteideas.net/book/the-book.html

	// Softmaxの時の温度パラメーター。
	// エンジンオプションの"Softmax_Temperature"で設定できる。

	/*
		softmax関数は⇓こう。
			softmax(x_i)   = exp(x_i) / Σ exp(x_j) for j
		ここに温度パラメーターTを導入。(これがSoftmax_Temparature)
			softmax_T(x_i) = exp(x_i / T) / Σ exp(x_j / T) for j
		そうすると分布の分散が変わる。
		Tが大きいとx_iの範囲が縮まるため、分散は下がる。
		Tが小さいと分散は広がる。

		探索で読み抜けする時は、Tを下げるように調整する。
	*/

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

		/*
		note :
			softmax関数の定義は⇓こうなので
				softmax(x_i) = exp(x_i) / Σ exp(x_j) for j
			x_i + cのように定数加算したところで
				softmax(x_i + c) = exp(x_i + c) / Σ exp(x_j + c) for j
								 = exp(x_i)exp(c) / Σ exp(x_j)exp(c) for j
			でexp(c)で約分できて
								 = softmax(x_i)
			となるので分布は変わらない。
		*/

		float sum = 0.0f;
		for (float& x : log_probabilities) {
			x = expf(x - max);
			sum += x;
		}
		// normalize
		for (float& x : log_probabilities) {
			x /= sum;
		}
	}

	Result init_model_paths()
	{
		std::vector<string> model_paths;
		for (int i = 1; i <= max_gpu ; ++i)
			model_paths.emplace_back(Options["DNN_Model" + std::to_string(i)]);

		string eval_dir = Options["EvalDir"];

		ModelPaths.clear();

		// ファイルが存在することが既知であるpath。
		// (一度調べたやつは記憶しておく)
		// モデルファイルはたかだか1,2個だと思うのでvectorで十分。
		static std::vector<std::string> checked_paths;

		// モデルファイル存在チェック
		bool is_err = false;
		for (int i = 0; i < max_gpu ; ++i) {
			if (model_paths[i] != "")
			{
				string path = Path::Combine(eval_dir, model_paths[i].c_str());
				if (std::find(checked_paths.begin(), checked_paths.end(), path) == checked_paths.end())
				{
					// 未チェックのやつなので調べる。
					std::ifstream ifs(path);
					if (!ifs.is_open()) {
						sync_cout << "Error! : " << path << " file not found" << sync_endl;
						is_err = true;
						break;
					}
					// 記憶しておく。
					checked_paths.push_back(path);
				}
				ModelPaths.push_back(path);
			}
			else {
				ModelPaths.push_back("");
			}
		}
		if (is_err)
			return ResultCode::FileNotFound;

		return ResultCode::Ok;
	}

	// エンジンオプションで設定されたモデルファイル名。
	// この返し値のvectorのsize() == max_gpuのはず。
	std::vector<std::string>* get_model_paths()
	{
		return &ModelPaths;
	}

	// 評価値から価値(勝率)に変換
	// スケールパラメータは、elmo_for_learnの勝率から調査した値
	// 何かの変換の時に必要になる。
	float cp_to_value(const Value score , const float eval_coef)
	{
		return 1.0f / (1.0f + expf(-(float)score / eval_coef));
	}

	// 価値(勝率)を評価値[cp]に変換。
	// USIではcp(centi-pawn)でやりとりするので、そのための変換に必要。
	// 	 eval_coef : 勝率を評価値に変換する時の定数。default = 756
	//
	// 返し値 :
	//   +29900は、評価値の最大値
	//   -29900は、評価値の最小値
	//   +30000,-30000は、(おそらく)詰みのスコア
	Value value_to_cp(const float score , const float eval_coef)
	{
		int cp;
		if (score == 1.0f)
			cp =  30000;
		else if (score == 0.0f)
			cp = -30000;
		else
		{
			cp = (int)(-logf(1.0f / score - 1.0f) * eval_coef);

			// 勝率がオーバーフローしてたらclampしておく。
			cp = std::clamp(cp, -29900, +29900);
		}

		return (Value)cp;
	}


} // namespace Eval::dlshogi

using namespace Eval::dlshogi;

namespace Eval
{
	void init(){}
	Value compute_eval(const Position& pos) { return VALUE_ZERO; }
	void evaluate_with_no_return(const Position& pos) {}
	void print_eval_stat(Position& pos) {}

	// 時間のかかる初期化処理はここに書く。
	// 毎回呼び出されるようになっている。
	void load_eval()
	{
		// 初回初期化。
		static bool init = false;
		if (!init)
		{
			// 指し手に対して、Policy Networkの返してくる配列のindexを返すテーブルの初期化
			dlshogi::init_move_label();
			init = true;
		}

		// モデルファイル(NNで使う評価関数ファイル)の確認。
		// これは"isready"に対して毎回初期化する。(ファイル名が変わるかも知れないので)
		dlshogi::init_model_paths();
	}

	// 考え中。
	// NN::forward()を呼ぶ実装にするかも。
	Value evaluate(const Position& pos) { return VALUE_ZERO; }
}

#endif // defined(EVAL_DEEP
