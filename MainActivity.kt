package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

class MainActivity : ComponentActivity() {

    private val viewModel: EquityHeatmapViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            // 为应用提供一个基础的暗色主题
            MaterialTheme(
                colorScheme = darkColorScheme(
                    background = Color(0xFF2E2E2E),
                    onBackground = Color.White,
                    surface = Color(0xFF2E2E2E),
                    onSurface = Color.White
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val uiState by viewModel.uiState.collectAsState()
                    EquityHeatmapScreen(
                        uiState = uiState,
                        onHandClick = { heroHand -> viewModel.displayHeatmap(heroHand) },
                        onResetClick = { viewModel.resetGridVisuals() }
                    )
                }
            }
        }
    }
}

@Composable
fun EquityHeatmapScreen(
    uiState: HeatmapUiState,
    onHandClick: (String) -> Unit,
    onResetClick: () -> Unit
) {
    val ranks = "AKQJT98765432"

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = uiState.title,
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // 主网格容器
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            // 顶部横坐标轴
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                // 为左侧纵坐标轴留出空间
                Spacer(modifier = Modifier.width(30.dp))
                ranks.forEach { rank ->
                    Text(
                        text = rank.toString(),
                        modifier = Modifier.weight(1f),
                        textAlign = TextAlign.Center,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // 包含左侧纵坐标轴和按钮的网格
            ranks.forEachIndexed { r_idx, r1 ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    // 左侧纵坐标轴标签
                    Text(
                        text = r1.toString(),
                        modifier = Modifier.width(30.dp),
                        textAlign = TextAlign.Center,
                        fontWeight = FontWeight.Bold
                    )

                    // 当前行的网格按钮
                    ranks.forEachIndexed { c_idx, r2 ->
                        val handKey = when {
                            r_idx < c_idx -> "${r1}${r2}s"
                            c_idx < r_idx -> "${ranks[c_idx]}${ranks[r_idx]}o"
                            else -> "$r1$r2"
                        }

                        val cellState = uiState.gridStates[handKey]

                        if (cellState != null) {
                            Button(
                                onClick = { onHandClick(cellState.hand) },
                                modifier = Modifier
                                    .weight(1f)
                                    .aspectRatio(1f) // 保证按钮是正方形
                                    .padding(1.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = cellState.backgroundColor,
                                    contentColor = cellState.textColor
                                ),
                                shape = MaterialTheme.shapes.extraSmall,
                                contentPadding = PaddingValues(0.dp)
                            ) {
                                Text(
                                    text = cellState.text,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    textAlign = TextAlign.Center,
                                    lineHeight = 12.sp
                                )
                            }
                        } else {
                            // 如果状态中没有对应的键，则留空，以保持布局对齐
                            Spacer(modifier = Modifier.weight(1f).aspectRatio(1f).padding(1.dp))
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(onClick = onResetClick) {
            Text("重置视图")
        }
    }
}

@Preview(showBackground = true, widthDp = 400, heightDp = 800)
@Composable
fun DefaultPreview() {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = Color(0xFF2E2E2E),
            onBackground = Color.White,
            surface = Color(0xFF2E2E2E),
            onSurface = Color.White
        )
    ) {
        // 创建一个用于预览的模拟UI状态
        val ranks = "AKQJT98765432"
        val suitedColor = Color(0xFF4A7A96)
        val offSuitColor = Color(0xFF4F4F4F)
        val pairColor = Color(0xFF8FBC8F)
        val initialGrid = mutableMapOf<String, GridCellState>()
        for (r_idx in ranks.indices) {
            for (c_idx in ranks.indices) {
                val r1 = ranks[r_idx]
                val r2 = ranks[c_idx]
                val (text, bgColor) = when {
                    r_idx < c_idx -> Pair("${r1}${r2}s", suitedColor)
                    c_idx < r_idx -> Pair("${ranks[c_idx]}${ranks[r_idx]}o", offSuitColor)
                    else -> Pair("$r1$r2", pairColor)
                }
                initialGrid[text] = GridCellState(hand = text, backgroundColor = bgColor)
            }
        }

        EquityHeatmapScreen(
            uiState = HeatmapUiState(gridStates = initialGrid),
            onHandClick = {},
            onResetClick = {}
        )
    }
}

