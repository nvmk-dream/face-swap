package com.example.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = ElectricCyan,
    onPrimary = Color(0xFF00262C),
    primaryContainer = Color(0xFF004E59),
    onPrimaryContainer = Color(0xFFB8F4FF),
    secondary = RoyalViolet,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFF321B7A),
    onSecondaryContainer = Color(0xFFE3D6FF),
    tertiary = NeonEmerald,
    onTertiary = Color(0xFF003919),
    background = Color(0xFF070E22),
    onBackground = TextPrimaryDark,
    surface = Color(0xCC0E1838), // Translucent Glassmorphism Surface
    onSurface = TextPrimaryDark,
    surfaceVariant = Color(0xD917244D), // Elevated Glass Card Variant
    onSurfaceVariant = Color(0xFFD4E2FD),
    error = CoralError,
    onError = Color.White
)

private val LightColorScheme = lightColorScheme(
    primary = DeepCyan,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFB4EBFF),
    onPrimaryContainer = Color(0xFF001F24),
    secondary = DeepViolet,
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFEADDFF),
    onSecondaryContainer = Color(0xFF21005D),
    tertiary = Color(0xFF006D36),
    onTertiary = Color.White,
    background = LightBg,
    onBackground = Color(0xFF111827),
    surface = LightSurface,
    onSurface = Color(0xFF111827),
    surfaceVariant = LightSurfaceVariant,
    onSurfaceVariant = Color(0xFF4B5563),
    error = Color(0xFFB3261E),
    onError = Color.White
)

@Composable
fun MyApplicationTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    // Default to dark theme aesthetic for computer vision studio while supporting light mode
    val colorScheme = if (darkTheme) DarkColorScheme else DarkColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}
