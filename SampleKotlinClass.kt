import java.util.*

/**
 * Sample Kotlin class that demonstrates basic structure.
 * This class provides a simple example of a data holder
 * and a basic method that operates on its values.
 */

class SampleKotlinClass(private val name: String, private val value: Int) {

    /**
     * Returns the current state of the instance as a string.
     */
    fun toStringRepresentation(): String {
        return "SampleKotlinClass(name='$name', value=$value)"
    }

    /**
     * Increments the internal value by the specified amount.
     * @param delta the amount to add
     */
    fun increment(delta: Int = 1) {
        // Use a local variable to preserve immutability of value
        val newValue = value + delta
        println("$name value changed from $value to $newValue")
    }
}

fun main() {
    val example = SampleKotlinClass("Test", 10)
    println(example.toStringRepresentation())
    example.increment(5)
    println(example.toStringRepresentation())
}
