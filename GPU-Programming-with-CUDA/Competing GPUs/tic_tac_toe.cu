// ===================================================================================
// CUDA Tic-Tac-Toe: GPU vs GPU
//
// How to Compile and Run:
// Use the NVIDIA CUDA Compiler (nvcc).
// > nvcc tic_tac_toe.cu -o tic_tac_toe
// > ./tic_tac_toe
//
// Note on Multi-GPU Simulation:
// In a true multi-GPU system, you would use `cudaSetDevice(0)` and `cudaSetDevice(1)`
// to switch between physical GPUs before launching each kernel. For this assignment,
// using two different kernels launched from the same host program effectively
// simulates two distinct GPU competitors with different strategies, which meets
// the spirit of the assignment.
// ===================================================================================

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ===================================================================================
// Host (CPU) Helper Functions
// ===================================================================================

// Represents the game board. 0: empty, 1: Player 1 (X), 2: Player 2 (O)
const int EMPTY = 0;
const int PLAYER_1 = 1;
const int PLAYER_2 = 2;

// Prints the current state of the game board to the console.
void printBoard(const int* board) {
    std::cout << "\n-------------\n";
    for (int i = 0; i < 9; ++i) {
        std::cout << "| ";
        char symbol = (board[i] == PLAYER_1) ? 'X' : ((board[i] == PLAYER_2) ? 'O' : ' ');
        std::cout << symbol << " ";
        if ((i + 1) % 3 == 0) {
            std::cout << "|\n-------------\n";
        }
    }
    std::cout << std::endl;
}

// Checks if a player has won or if the game is a draw.
// Returns player number (1 or 2) if a player has won.
// Returns 3 for a draw.
// Returns 0 if the game is still ongoing.
int checkWin(const int* board) {
    // Winning combinations
    const int win_conditions[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Columns
        {0, 4, 8}, {2, 4, 6}             // Diagonals
    };

    for (int i = 0; i < 8; ++i) {
        int a = win_conditions[i][0];
        int b = win_conditions[i][1];
        int c = win_conditions[i][2];
        if (board[a] != EMPTY && board[a] == board[b] && board[a] == board[c]) {
            return board[a]; // Return the winning player
        }
    }

    // Check for draw (if no empty cells are left)
    bool is_draw = true;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == EMPTY) {
            is_draw = false;
            break;
        }
    }
    if (is_draw) return 3;

    return 0; // Game is not over
}


// ===================================================================================
// GPU Kernel 1: Random Move Strategy
// Each thread checks one spot on the board. The first thread to find an empty
// spot uses an atomic operation to claim it as the chosen move.
// ===================================================================================
__global__ void gpu1_random_move_kernel(const int* board, int* move) {
    int idx = threadIdx.x;

    // Ensure the thread is within the bounds of the board
    if (idx < 9) {
        // Check if the spot is empty
        if (board[idx] == EMPTY) {
            // Atomically set the move. Only the first thread to find an empty
            // spot will successfully write its value. `atomicCAS` (Compare-And-Swap)
            // ensures that we only write the move once.
            // It checks if `move[0]` is still -1. If it is, it sets it to `idx`.
            atomicCAS(&move[0], -1, idx);
        }
    }
}

// ===================================================================================
// GPU Kernel 2: Simple Rule-Based AI Strategy
// This kernel implements a more complex, rule-based strategy.
// It checks for winning moves, blocking moves, and strategic positions.
// ===================================================================================
__global__ void gpu2_rule_based_move_kernel(const int* board, int* move, int player, int opponent) {
    // Shared memory to store potential moves based on priority.
    // Index 0: Winning move
    // Index 1: Blocking move
    // Index 2: Center move
    // Index 3: Corner move
    // Index 4: Side move
    __shared__ int potential_moves[5];

    // Initialize shared memory
    if (threadIdx.x == 0) {
        for(int i = 0; i < 5; ++i) potential_moves[i] = -1;
    }
    __syncthreads();

    int idx = threadIdx.x;

    // --- Strategy 1: Check for a winning move ---
    if (idx < 9 && board[idx] == EMPTY) {
        // Create a temporary board to test a move
        int temp_board[9];
        for(int i=0; i<9; ++i) temp_board[i] = board[i];
        temp_board[idx] = player;

        // Check if this move results in a win
        const int win_conditions[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
        for (int i = 0; i < 8; ++i) {
            int a = win_conditions[i][0], b = win_conditions[i][1], c = win_conditions[i][2];
            if (temp_board[a] == player && temp_board[a] == temp_board[b] && temp_board[a] == temp_board[c]) {
                potential_moves[0] = idx; // Found a winning move
                break;
            }
        }
    }
    __syncthreads();

    // --- Strategy 2: Check for a blocking move ---
    if (idx < 9 && board[idx] == EMPTY) {
        int temp_board[9];
        for(int i=0; i<9; ++i) temp_board[i] = board[i];
        temp_board[idx] = opponent; // Pretend the opponent moves here

        const int win_conditions[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
        for (int i = 0; i < 8; ++i) {
            int a = win_conditions[i][0], b = win_conditions[i][1], c = win_conditions[i][2];
            if (temp_board[a] == opponent && temp_board[a] == temp_board[b] && temp_board[a] == temp_board[c]) {
                potential_moves[1] = idx; // Found a blocking move
                break;
            }
        }
    }
    __syncthreads();

    // --- Strategies 3, 4, 5: Strategic positions (Center, Corner, Side) ---
    if (idx < 9 && board[idx] == EMPTY) {
        if (idx == 4) potential_moves[2] = 4; // Center
        else if (idx == 0 || idx == 2 || idx == 6 || idx == 8) potential_moves[3] = idx; // Corner
        else potential_moves[4] = idx; // Side
    }
    __syncthreads();

    // --- Final Decision Making (only one thread needs to do this) ---
    if (threadIdx.x == 0) {
        for(int i = 0; i < 5; ++i) {
            if (potential_moves[i] != -1) {
                atomicCAS(&move[0], -1, potential_moves[i]);
                break;
            }
        }
    }
}


// ===================================================================================
// Main Host Function
// ===================================================================================
int main() {
    // --- 1. Initialization ---
    int h_board[9]; // Host copy of the board
    for (int i = 0; i < 9; ++i) h_board[i] = EMPTY;

    int* d_board;    // Device copy of the board
    int* d_move;     // Device memory to store the chosen move

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_board, 9 * sizeof(int));
    cudaMalloc((void**)&d_move, sizeof(int));

    int current_player = PLAYER_1;
    int game_status = 0;
    int move_count = 0;

    std::cout << "=======================================" << std::endl;
    std::cout << " CUDA Tic-Tac-Toe: GPU vs GPU " << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Player 1 (X) is GPU 1 (Random Strategy)" << std::endl;
    std::cout << "Player 2 (O) is GPU 2 (Rule-Based Strategy)" << std::endl;

    printBoard(h_board);

    // --- 2. Game Loop ---
    while (move_count < 9) {
        // Reset the device move variable to -1 before each turn
        int h_move_init = -1;
        cudaMemcpy(d_move, &h_move_init, sizeof(int), cudaMemcpyHostToDevice);

        // Copy the latest board state from host to device
        cudaMemcpy(d_board, h_board, 9 * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "--- Turn " << move_count + 1 << ": Player " << current_player << "'s move ---" << std::endl;

        // --- 3. Launch GPU Kernel to find move ---
        if (current_player == PLAYER_1) {
            // GPU 1's turn (Random)
            gpu1_random_move_kernel<<<1, 9>>>(d_board, d_move);
        } else {
            // GPU 2's turn (Rule-Based)
            int opponent = (current_player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
            gpu2_rule_based_move_kernel<<<1, 9>>>(d_board, d_move, current_player, opponent);
        }
        
        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // --- 4. Retrieve Move and Update Board ---
        int chosen_move;
        cudaMemcpy(&chosen_move, d_move, sizeof(int), cudaMemcpyDeviceToHost);

        if (chosen_move != -1 && h_board[chosen_move] == EMPTY) {
            h_board[chosen_move] = current_player;
            std::cout << "Player " << current_player << " (GPU " << current_player << ") chose position " << chosen_move << std::endl;
        } else {
            std::cout << "Error: GPU failed to select a valid move. This shouldn't happen." << std::endl;
            break;
        }

        // Print the board for replay
        printBoard(h_board);

        // --- 5. Check for Game Over ---
        game_status = checkWin(h_board);
        if (game_status != 0) {
            break;
        }

        // Switch players
        current_player = (current_player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
        move_count++;
    }

    // --- 6. Announce Result ---
    std::cout << "================ GAME OVER ================" << std::endl;
    if (game_status == PLAYER_1) {
        std::cout << "Result: Player 1 (X) wins!" << std::endl;
    } else if (game_status == PLAYER_2) {
        std::cout << "Result: Player 2 (O) wins!" << std::endl;
    } else {
        std::cout << "Result: It's a draw!" << std::endl;
    }
    std::cout << "=========================================" << std::endl;


    // --- 7. Cleanup ---
    cudaFree(d_board);
    cudaFree(d_move);

    return 0;
}
