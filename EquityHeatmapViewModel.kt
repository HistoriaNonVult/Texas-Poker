package com.example.myapplication

import android.app.Application
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.InputStreamReader

// 定义网格中每个单元格的可视化状态
data class GridCellState(
    val hand: String,
    val text: String = hand,
    val backgroundColor: Color,
    val textColor: Color = Color.White
)

// 定义整个UI界面的状态
data class HeatmapUiState(
    val title: String = "请在下方网格中点击一手牌开始分析",
    val gridStates: Map<String, GridCellState> = emptyMap()
)

class EquityHeatmapViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(HeatmapUiState())
    val uiState: StateFlow<HeatmapUiState> = _uiState.asStateFlow()

    private val ranks = "AKQJT98765432"
    private var equityDatabase: Map<String, Map<String, Double>>? = null

    // 为初始网格状态预定义颜色
    private val suitedColor = Color(0xFF4A7A96)
    private val offSuitColor = Color(0xFF4F4F4F)
    private val pairColor = Color(0xFF8FBC8F)

    init {
        // 在ViewModel初始化时，异步加载数据库并设置初始网格
        viewModelScope.launch {
            loadDatabase()
            resetGridVisuals()
        }
    }

    private fun loadDatabase() {
        try {
            val context = getApplication<Application>().applicationContext
            context.assets.open("full_equity_database.json").use { inputStream ->
                InputStreamReader(inputStream).use { reader ->
                    val type = object : TypeToken<Map<String, Map<String, Double>>>() {}.type
                    equityDatabase = Gson().fromJson(reader, type)
                    _uiState.update { it.copy(title = "请在下方网格中点击一手牌开始分析") }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            _uiState.update {
                it.copy(title = "错误: 未找到 'full_equity_database.json'")
            }
        }
    }

    fun displayHeatmap(heroHand: String) {
        val db = equityDatabase
        if (db == null) {
            _uiState.update { it.copy(title = "数据库未加载，无法显示热力图。") }
            return
        }

        val results = db[heroHand]
        if (results == null) {
            _uiState.update { it.copy(title = "在数据库中未找到手牌 '$heroHand' 的数据。") }
            return
        }

        // 在应用新的热力图前，先重置视图
        resetGridVisuals()

        _uiState.update { it.copy(title = "Hero: $heroHand") }

        val newGridStates = _uiState.value.gridStates.toMutableMap()

        results.forEach { (villainHand, equity) ->
            val currentCell = newGridStates[villainHand]
            if (currentCell != null) {
                val bgColor = getColorForEquity(equity)
                val textColor = getTextColorForBg(bgColor)
                newGridStates[villainHand] = currentCell.copy(
                    backgroundColor = bgColor,
                    textColor = textColor,
                    text = "%.1f%%".format(equity)
                )
            }
        }

        // 标记英雄手牌
        val heroCell = newGridStates[heroHand]
        if (heroCell != null) {
            newGridStates[heroHand] = heroCell.copy(
                backgroundColor = Color(0xFFFFD700), //金色
                textColor = Color.Black,
                text = "HERO"
            )
        }

        _uiState.update { it.copy(gridStates = newGridStates) }
    }


    fun resetGridVisuals() {
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
        _uiState.update {
            it.copy(
                gridStates = initialGrid,
                title = if (equityDatabase == null) "错误: 未找到 'full_equity_database.json'" else "请在下方网格中点击一手牌开始分析"
            )
        }
    }

    private fun getColorForEquity(equity: Double): Color {
        val e = (equity / 100.0).coerceIn(0.0, 1.0)
        val blue = Triple(23, 92, 201)
        val yellow = Triple(255, 255, 224)
        val red = Triple(214, 40, 40)

        val (r, g, b) = if (e < 0.5) {
            val t = e * 2
            Triple(
                (blue.first * (1 - t) + yellow.first * t).toInt(),
                (blue.second * (1 - t) + yellow.second * t).toInt(),
                (blue.third * (1 - t) + yellow.third * t).toInt()
            )
        } else {
            val t = (e - 0.5) * 2
            Triple(
                (yellow.first * (1 - t) + red.first * t).toInt(),
                (yellow.second * (1 - t) + red.second * t).toInt(),
                (yellow.third * (1 - t) + red.third * t).toInt()
            )
        }
        return Color(r, g, b)
    }

    private fun getTextColorForBg(backgroundColor: Color): Color {
        val r = backgroundColor.red * 255
        val g = backgroundColor.green * 255
        val b = backgroundColor.blue * 255
        return if ((0.299 * r + 0.587 * g + 0.114 * b) > 150) Color.Black else Color.White
    }
}

